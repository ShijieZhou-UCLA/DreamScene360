
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import glob
import importlib
import inspect
import time
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import argparse
import math
import os
import random
import re

import diffusers
import numpy as np
import torch
import torchvision
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from tqdm import tqdm
from torchvision import transforms
transformers = importlib.import_module('transformers')

CLIPTextModel = getattr(transformers, 'CLIPTextModel')
CLIPTokenizer = getattr(transformers, 'CLIPTokenizer')
CLIPModel = getattr(transformers, 'CLIPModel')
CLIPTextConfig = getattr(transformers, 'CLIPTextConfig')
print("Transformers version:", transformers.__version__)

#from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPTextConfig
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import library.model_util as model_util
import library.train_util as train_util
from networks.lora import LoRANetwork

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# other settings
LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8
FEATURE_EXTRACTOR_SIZE = (224, 224)
FEATURE_EXTRACTOR_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FEATURE_EXTRACTOR_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


## following MultiDiffusion: https://github.com/omerbt/MultiDiffusion/blob/master/panorama.py ##
## the window size is changed for 360-degree panorama generation ##
def get_views(panorama_height, panorama_width, window_size=[64,128], stride=16):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size[0]) // stride + 1
    num_blocks_width = (panorama_width - window_size[1]) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size[0]
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size[1]
        views.append((h_start, h_end, w_start, w_end))
    return views


class PipelineLike:
    r"""
    Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
    weighting in prompt.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        device,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        clip_skip: int,
    ):
        super().__init__()
        self.device = device
        self.clip_skip = clip_skip

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

    def replace_token(self, tokens, layer=None):
        new_tokens = []
        for token in tokens:
            new_tokens.append(token)
        return new_tokens

    def enable_xformers_memory_efficient_attention(self):
        r"""
        Enable memory efficient attention as implemented in xformers.
        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.
        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.
        """
        self.unet.set_use_memory_efficient_attention_xformers(True)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_scale: float = None,
        strength: float = 0.8,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        vae_batch_size: float = None,
        return_latents: bool = False,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: Optional[int] = 1,
        img2img_noise=None,
        clip_prompts=None,
        clip_guide_images=None,
        networks: Optional[List[LoRANetwork]] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        num_images_per_prompt = 1  # fixed

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        vae_batch_size = (
            batch_size
            if vae_batch_size is None
            else (int(vae_batch_size) if vae_batch_size >= 1 else max(1, int(batch_size * vae_batch_size)))
        )

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}."
            )

        # get prompt text embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if not do_classifier_free_guidance and negative_scale is not None:
            print(f"negative_scale is ignored if guidance scalle <= 1.0")
            negative_scale = None

        # get unconditional embeddings for classifier free guidance
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
            pipe=self,
            prompt=prompt,
            uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
            max_embeddings_multiples=max_embeddings_multiples,
            clip_skip=self.clip_skip,
            **kwargs,
        )

        if do_classifier_free_guidance:
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)

        latents_dtype = text_embeddings.dtype

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (
            batch_size * num_images_per_prompt,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )

        if latents is None:
            if self.device.type == "mps":
                # randn does not exist on mps
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device="cpu",
                    dtype=latents_dtype,
                ).to(self.device)
            else:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=self.device,
                    dtype=latents_dtype,
                )
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        timesteps = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        num_latent_input = (3 if negative_scale is not None else 2) if do_classifier_free_guidance else 1

        #####################
        ## StitchDiffusion ##
        #####################

        views_t = get_views(height, width)
        count_t = torch.zeros_like(latents)
        value_t = torch.zeros_like(latents)

        for i, t in enumerate(tqdm(timesteps)):

            count_t.zero_()
            value_t.zero_()

            # initialize the value of latent_view_t
            latent_view_t = latents[:, :, :, 64:192]

            # pre-denoising operations twice on the stitch block
            for ii_md in range(2):

                latent_view_t[:, :, :, 0:64] = latents[:, :, :, 192:256]
                latent_view_t[:, :, :, 64:128] = latents[:, :, :, 0:64]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = latent_view_t.repeat((num_latent_input, 1, 1, 1))
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latent_view_denoised = self.scheduler.step(noise_pred, t, latent_view_t, **extra_step_kwargs).prev_sample

                value_t[:, :, :, 192:256] += latent_view_denoised[:, :, :, 0:64]
                count_t[:, :, :, 192:256] += 1

                value_t[:, :, :, 0:64] += latent_view_denoised[:, :, :, 64:128]
                count_t[:, :, :, 0:64] += 1

            # same denoising operations as what MultiDiffusion does
            for h_start, h_end, w_start, w_end in views_t:

                latent_view_t = latents[:, :, h_start:h_end, w_start:w_end]
            
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latent_view_t.repeat((num_latent_input, 1, 1, 1))
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latent_view_denoised = self.scheduler.step(noise_pred, t, latent_view_t, **extra_step_kwargs).prev_sample
                value_t[:, :, h_start:h_end, w_start:w_end] += latent_view_denoised
                count_t[:, :, h_start:h_end, w_start:w_end] += 1

            latents = torch.where(count_t > 0, value_t / count_t, value_t)

        if return_latents:
            return (latents, False)

        latents = 1 / 0.18215 * latents
        if vae_batch_size >= batch_size:
            image = self.vae.decode(latents).sample
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            images = []
            for i in tqdm(range(0, batch_size, vae_batch_size)):
                images.append(
                    self.vae.decode(latents[i : i + vae_batch_size] if vae_batch_size > 1 else latents[i].unsqueeze(0)).sample
                )
            image = torch.cat(images)

        image = (image / 2 + 0.5).clamp(0, 1)

        # global cropping operation
        image = image[:, :, :, 512:1536]

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        has_nsfw_concept = None

        if output_type == "pil":
            # image = self.numpy_to_pil(image)
            image = (image * 255).round().astype("uint8")
            image = [Image.fromarray(im) for im in image]

        # if not return_dict:
        return (image, has_nsfw_concept)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe: PipelineLike, prompt: List[str], max_length: int, layer=None):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]

            token = pipe.replace_token(token, layer=layer)

            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        print("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: PipelineLike,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = pipe.text_encoder(text_input_chunk)[0]
            else:
                enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = pipe.text_encoder(text_input)[0]
        else:
            enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    pipe: PipelineLike,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
    layer=None,
    **kwargs,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `1`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    # split the prompts with "AND". each prompt must have the same number of splits
    new_prompts = []
    for p in prompt:
        new_prompts.extend(p.split(" AND "))
    prompt = new_prompts

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2, layer=layer)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2, layer=layer)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = pipe.tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

    # assign weights to the prompts and normalize in the sense of mean
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings, prompt_tokens
    return text_embeddings, None, prompt_tokens


class BatchDataBase(NamedTuple):
    # data without batch partitioning
    step: int
    prompt: str
    negative_prompt: str
    seed: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any


class BatchDataExt(NamedTuple):
    # batch partitioning data
    width: int
    height: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Tuple[float]
    num_sub_prompts: int


class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt


def main(args):
    if args.fp16:
        dtype = torch.float16
    else:
        # Stop the program if the condition is not satisfied
        print("precision is not satisfied. Program stopped.")
        assert args.fp16, "precision is not satisfied."

    highres_fix = args.highres_fix_scale is not None
    assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img"

    if args.v_parameterization and not args.v2:
        print("v_parameterization should be with v2")
    if args.v2 and args.clip_skip is not None:
        print("v2 with clip_skip will be unexpected")

    # Load model
    if not os.path.isfile(args.ckpt):  #
        files = glob.glob(args.ckpt)
        if len(files) == 1:
            args.ckpt = files[0]

    use_stable_diffusion_format = os.path.isfile(args.ckpt)
    print('Stable Diffusion ckpt path:',args.ckpt)
    if use_stable_diffusion_format:
        print("load StableDiffusion checkpoint")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.ckpt)
    else:
        # Stop the program if the condition is not satisfied
        print("precision is not satisfied. Program stopped.")
        assert use_stable_diffusion_format, "precision is not satisfied."


    # Load VAE
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, dtype)
        print("additional VAE loaded")

    # clip_model = None

    # Load tokenizer
    print("loading tokenizer")
    if use_stable_diffusion_format:
        tokenizer = train_util.load_tokenizer(args)

    # scheduler preparation
    sched_init_args = {}
    scheduler_num_noises_per_step = 1
    if args.sampler == "ddim":
        scheduler_cls = DDIMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddim
    else:
        # Stop the program if the condition is not satisfied
        print("sampler is not satisfied. Program stopped.")
        assert args.sampler, "sampler is not satisfied."

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    # processing to specify sampler's random number

    # replace randn
    class NoiseManager:
        def __init__(self):
            self.sampler_noises = None
            self.sampler_noise_index = 0

        def reset_sampler_noises(self, noises):
            self.sampler_noise_index = 0
            self.sampler_noises = noises

        def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
            # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
            if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
                noise = self.sampler_noises[self.sampler_noise_index]
                if shape != noise.shape:
                    noise = None
            else:
                noise = None

            if noise == None:
                print(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
                noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

            self.sampler_noise_index += 1
            return noise

    class TorchRandReplacer:
        def __init__(self, noise_manager):
            self.noise_manager = noise_manager

        def __getattr__(self, item):
            if item == "randn":
                return self.noise_manager.randn
            if hasattr(torch, item):
                return getattr(torch, item)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    noise_manager = NoiseManager()
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        print("set clip_sample to True")
        scheduler.config.clip_sample = True

    # assign device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # create a custom pipeline
    vae.to(dtype).to(device)
    text_encoder.to(dtype).to(device)
    unet.to(dtype).to(device)

    # incorporating LoRA
    print(args.network_module)
    if args.network_module:
        networks = []
        network_default_muls = []
        for i, network_module in enumerate(args.network_module):
            print("import network module:", network_module)
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]
            network_default_muls.append(network_mul)

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i]
                # TODO escape special chars
                network_args = network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if args.network_weights and i < len(args.network_weights):
                network_weight = args.network_weights[i]
                print("load network weights from:", network_weight)

                if model_util.is_safetensors(network_weight) and args.network_show_meta:
                    from safetensors.torch import safe_open

                    with safe_open(network_weight, framework="pt") as f:
                        metadata = f.metadata()
                    if metadata is not None:
                        print(f"metadata for: {network_weight}: {metadata}")

                network, weights_sd = imported_module.create_network_from_weights(
                    network_mul, network_weight, vae, text_encoder, unet, for_inference=True, **net_kwargs
                )
            else:
                raise ValueError("No weight. Weight is required.")
            if network is None:
                return

            mergiable = hasattr(network, "merge_to")
            if args.network_merge and not mergiable:
                print("network is not mergiable. ignore merge option.")

            if not args.network_merge or not mergiable:
                network.apply_to(text_encoder, unet)
                info = network.load_state_dict(weights_sd, False)
                print(f"weights are loaded: {info}")

                if args.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(dtype).to(device)

                networks.append(network)
            else:
                network.merge_to(text_encoder, unet, weights_sd, dtype, device)

    else:
        networks = []

    pipe = PipelineLike(
        device,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        args.clip_skip,
    )
    # pdb.set_trace()
    print("pipeline is ready.")

    if args.diffusers_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    # prompt acquisition
    if args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = []

    if args.interactive:
        args.n_iter = 1

    regional_network = False

    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        predefined_seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(args.n_iter * len(prompt_list) * args.images_per_prompt)]
        if len(predefined_seeds) == 1:
            predefined_seeds[0] = args.seed
    else:
        predefined_seeds = None

    # set default image size
    if args.W is None:
        args.W = 512
    if args.H is None:
        args.H = 512

    # image generation loops
    os.makedirs(args.outdir, exist_ok=True)
    max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

    for gen_iter in range(args.n_iter):
        print(f"iteration {gen_iter+1}/{args.n_iter}")
        iter_seed = random.randint(0, 0x7FFFFFFF)

        # batch processing functions
        def process_batch(batch: List[BatchData], highres_fix, highres_1st=False):
            batch_size = len(batch)

            # extract batch information
            (
                return_latents,
                (step_first, _, _, _, init_image, mask_image, _, guide_image),
                (width, height, steps, scale, negative_scale, strength, network_muls, num_sub_prompts),
            ) = batch[0]
            noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

            prompts = []
            negative_prompts = []
            start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
            noises = [
                torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                for _ in range(steps * scheduler_num_noises_per_step)
            ]
            seeds = []
            clip_prompts = []

            i2i_noises = None
            init_images = None
            mask_images = None

            if guide_image is not None:  # CLIP image guided?
                guide_images = []
            else:
                guide_images = None

            # Random numbers are generated here to use the same random numbers regardless of the location in the batch. 
            # Check whether image / mask is identical in batch
            all_images_are_same = True
            all_masks_are_same = True
            all_guide_images_are_same = True
            for i, (_, (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image), _) in enumerate(batch):
                prompts.append(prompt)
                negative_prompts.append(negative_prompt)
                seeds.append(seed)
                clip_prompts.append(clip_prompt)

                if init_image is not None:
                    init_images.append(init_image)
                    if i > 0 and all_images_are_same:
                        all_images_are_same = init_images[-2] is init_image

                if mask_image is not None:
                    mask_images.append(mask_image)
                    if i > 0 and all_masks_are_same:
                        all_masks_are_same = mask_images[-2] is mask_image

                if guide_image is not None:
                    if type(guide_image) is list:
                        guide_images.extend(guide_image)
                        all_guide_images_are_same = False
                    else:
                        guide_images.append(guide_image)
                        if i > 0 and all_guide_images_are_same:
                            all_guide_images_are_same = guide_images[-2] is guide_image

                # make start code
                torch.manual_seed(seed)
                start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

                # make each noises
                for j in range(steps * scheduler_num_noises_per_step):
                    noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

                if i2i_noises is not None:  # img2img noise
                    i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

            noise_manager.reset_sampler_noises(noises)

            # If all images are the same, pass the pipe to one pipe to speed up the pipe
            if init_images is not None and all_images_are_same:
                init_images = init_images[0]
            if mask_images is not None and all_masks_are_same:
                mask_images = mask_images[0]
            if guide_images is not None and all_guide_images_are_same:
                guide_images = guide_images[0]

            # generate
            if networks:
                shared = {}
                for n, m in zip(networks, network_muls if network_muls else network_default_muls):
                    n.set_multiplier(m)
                    if regional_network:
                        n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

            images = pipe(
                prompts,
                negative_prompts,
                init_images,
                mask_images,
                height,
                width,
                steps,
                scale,
                negative_scale,
                strength,
                latents=start_code,
                output_type="pil",
                max_embeddings_multiples=max_embeddings_multiples,
                img2img_noise=i2i_noises,
                vae_batch_size=args.vae_batch_size,
                return_latents=return_latents,
                clip_prompts=clip_prompts,
                clip_guide_images=guide_images,
            )[0]
            if highres_1st and not args.highres_fix_save_1st:  # return images or latents
                return images

            # save image
            highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            for i, (image, prompt, negative_prompts, seed, clip_prompt) in enumerate(
                zip(images, prompts, negative_prompts, seeds, clip_prompts)
            ):
                metadata = PngInfo()
                metadata.add_text("prompt", prompt)
                metadata.add_text("seed", str(seed))
                metadata.add_text("sampler", args.sampler)
                metadata.add_text("steps", str(steps))
                metadata.add_text("scale", str(scale))
                if negative_prompt is not None:
                    metadata.add_text("negative-prompt", negative_prompt)
                if negative_scale is not None:
                    metadata.add_text("negative-scale", str(negative_scale))
                if clip_prompt is not None:
                    metadata.add_text("clip-prompt", clip_prompt)

                fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

                image.save(os.path.join(args.outdir, fln), pnginfo=metadata)

            if not args.no_preview and not highres_1st and args.interactive:
                try:
                    import cv2

                    for prompt, image in zip(prompts, images):
                        cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                except ImportError:
                    print("opencv-python is not installed, cannot preview")

            return images

        # loop until prompt for image generation 
        prompt_index = 0
        global_step = 0
        batch_data = []
        while args.interactive or prompt_index < len(prompt_list):
            if len(prompt_list) == 0:
                # interactive
                valid = False
                while not valid:
                    print("\nType prompt:")
                    try:
                        prompt = input()
                    except EOFError:
                        break

                    valid = len(prompt.strip().split(" --")[0].strip()) > 0
                if not valid:  # EOF, end app
                    break
            else:
                prompt = prompt_list[prompt_index]

            # parse prompt
            width = args.W
            height = args.H
            scale = args.scale
            negative_scale = args.negative_scale
            steps = args.steps
            seeds = None
            strength = 0.8 if args.strength is None else args.strength
            negative_prompt = ""
            clip_prompt = None
            network_muls = None

            prompt_args = prompt.strip().split(" --")
            prompt = prompt_args[0]
            print(f"prompt {prompt_index+1}/{len(prompt_list)}: {prompt}")

            for parg in prompt_args[1:]:
                try:
                    m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                    if m:
                        width = int(m.group(1))
                        print(f"width: {width}")
                        continue

                    m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                    if m:
                        height = int(m.group(1))
                        print(f"height: {height}")
                        continue

                    m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                    if m:  # steps
                        steps = max(1, min(1000, int(m.group(1))))
                        print(f"steps: {steps}")
                        continue

                    m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                    if m:  # seed
                        seeds = [int(d) for d in m.group(1).split(",")]
                        print(f"seeds: {seeds}")
                        continue

                    m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                    if m:  # scale
                        scale = float(m.group(1))
                        print(f"scale: {scale}")
                        continue

                    m = re.match(r"nl ([\d\.]+|none|None)", parg, re.IGNORECASE)
                    if m:  # negative scale
                        if m.group(1).lower() == "none":
                            negative_scale = None
                        else:
                            negative_scale = float(m.group(1))
                        print(f"negative scale: {negative_scale}")
                        continue

                    m = re.match(r"t ([\d\.]+)", parg, re.IGNORECASE)
                    if m:  # strength
                        strength = float(m.group(1))
                        print(f"strength: {strength}")
                        continue

                    m = re.match(r"n (.+)", parg, re.IGNORECASE)
                    if m:  # negative prompt
                        negative_prompt = m.group(1)
                        print(f"negative prompt: {negative_prompt}")
                        continue

                    m = re.match(r"c (.+)", parg, re.IGNORECASE)
                    if m:  # clip prompt
                        clip_prompt = m.group(1)
                        print(f"clip prompt: {clip_prompt}")
                        continue

                    m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
                    if m:  # network multiplies
                        network_muls = [float(v) for v in m.group(1).split(",")]
                        while len(network_muls) < len(networks):
                            network_muls.append(network_muls[-1])
                        print(f"network mul: {network_muls}")
                        continue

                except ValueError as ex:
                    print(f"Exception in parsing: {parg}")
                    print(ex)

            if seeds is not None:
                # repeat if repeat is insufficient
                if len(seeds) < args.images_per_prompt:
                    seeds = seeds * int(math.ceil(args.images_per_prompt / len(seeds)))
                seeds = seeds[: args.images_per_prompt]
            else:
                if predefined_seeds is not None:
                    seeds = predefined_seeds[-args.images_per_prompt :]
                    predefined_seeds = predefined_seeds[: -args.images_per_prompt]
                elif args.iter_same_seed:
                    seeds = [iter_seed] * args.images_per_prompt
                else:
                    seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(args.images_per_prompt)]
                if args.interactive:
                    print(f"seed: {seeds}")

            init_image = mask_image = guide_image = None
            for seed in seeds:

                if regional_network:
                    num_sub_prompts = len(prompt.split(" AND "))
                    assert (
                        len(networks) <= num_sub_prompts
                    ), "Number of networks must be less than or equal to number of sub prompts."
                else:
                    num_sub_prompts = None

                b1 = BatchData(
                    False,
                    BatchDataBase(global_step, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image),
                    BatchDataExt(
                        width,
                        height,
                        steps,
                        scale,
                        negative_scale,
                        strength,
                        tuple(network_muls) if network_muls else None,
                        num_sub_prompts,
                    ),
                )
                return( process_batch([b1], highres_fix))
                if len(batch_data) > 0 and batch_data[-1].ext != b1.ext:
                    return process_batch(batch_data, highres_fix)
                    batch_data.clear()

                batch_data.append(b1)
                if len(batch_data) == args.batch_size:
                    prev_image = process_batch(batch_data, highres_fix)[0]
                    batch_data.clear()

                global_step += 1

            prompt_index += 1

        if len(batch_data) > 0:
            return process_batch(batch_data, highres_fix)
            batch_data.clear()

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.0 model")
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training"
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt")
    parser.add_argument(
        "--interactive", action="store_true", help="interactive mode (generates one image)"
    )
    parser.add_argument(
        "--no_preview", action="store_true", help="do not show generated image in interactive mode"
    )
    parser.add_argument("--strength", type=float, default=None, help="img2img strength")
    parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--sequential_file_name", action="store_true", help="sequential output file name")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations")
    parser.add_argument("--H", type=int, default=None, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=None, help="image width, in pixel space")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--vae_batch_size",
        type=float,
        default=None,
        help="batch size for VAE, < 1.0 for ratio",
    )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
        ],
        help=f"sampler (scheduler) type",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint of model")
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace"
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed, or seed of seeds in multiple generation",
    )
    parser.add_argument(
        "--iter_same_seed",
        action="store_true",
        help="use same seed for all prompts in iteration if no seed specified",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--xformers", action="store_true", help="use xformers")
    parser.add_argument(
        "--diffusers_xformers",
        action="store_true",
        help="use xformers by diffusers (Hypernetworks doesn't work)",
    )
    parser.add_argument(
        "--opt_channels_last", action="store_true", help="set channels last option to model"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, nargs="*", help="additional network module to use"
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, nargs="*", help="additional network weights to load"
    )
    parser.add_argument("--network_mul", type=float, default=None, nargs="*", help="additional network multiplier")
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*", help="additional argmuments for network (key=value)"
    )
    parser.add_argument("--network_show_meta", action="store_true", help="show metadata of network model")
    parser.add_argument("--network_merge", action="store_true", help="merge network weights to original model")
    parser.add_argument("--clip_skip", type=int, default=None, help="layer number from bottom to use in CLIP")
    parser.add_argument(
        "--max_embeddings_multiples",
        type=int,
        default=None,
        help="max embeding multiples, max token length is 75 * multiples",
    )
    parser.add_argument(
        "--highres_fix_scale",
        type=float,
        default=None,
        help="enable highres fix, reso scale for 1st stage",
    )
    parser.add_argument(
        "--highres_fix_save_1st", action="store_true", help="save 1st stage images for highres fix"
    )
    parser.add_argument(
        "--negative_scale", type=float, default=None, help="set another guidance scale for negative prompt"
    )

    return parser


class StitchDiffusionArgs:
    def __init__(
        self,
        v2=False,
        v_parameterization=False,
        prompt=None,
        interactive=False,
        no_preview=False,
        strength=None,
        images_per_prompt=1,
        outdir="outputs",
        sequential_file_name=False,
        n_iter=1,
        H=None,
        W=None,
        batch_size=1,
        vae_batch_size=None,
        steps=50,
        sampler="ddim",
        scale=7.5,
        ckpt=None,
        vae=None,
        tokenizer_cache_dir=None,
        seed=None,
        iter_same_seed=False,
        fp16=False,
        xformers=False,
        diffusers_xformers=False,
        opt_channels_last=False,
        network_module=None,
        network_weights=None,
        network_mul=None,
        network_args=None,
        network_show_meta=False,
        network_merge=False,
        clip_skip=None,
        max_embeddings_multiples=None,
        highres_fix_scale=None,
        highres_fix_save_1st=False,
        negative_scale=None
    ):
        self.v2 = v2
        self.v_parameterization = v_parameterization
        self.prompt = prompt
        self.interactive = interactive
        self.no_preview = no_preview
        self.strength = strength
        self.images_per_prompt = images_per_prompt
        self.outdir = outdir
        self.sequential_file_name = sequential_file_name
        self.n_iter = n_iter
        self.H = H
        self.W = W
        self.batch_size = batch_size
        self.vae_batch_size = vae_batch_size
        self.steps = steps
        self.sampler = sampler
        self.scale = scale
        self.ckpt = ckpt
        self.vae = vae
        self.tokenizer_cache_dir = tokenizer_cache_dir
        self.seed = seed
        self.iter_same_seed = iter_same_seed
        self.fp16 = fp16
        self.xformers = xformers
        self.diffusers_xformers = diffusers_xformers
        self.opt_channels_last = opt_channels_last
        self.network_module = network_module
        self.network_weights = network_weights
        self.network_mul = network_mul
        self.network_args = network_args
        self.network_show_meta = network_show_meta
        self.network_merge = network_merge
        self.clip_skip = clip_skip
        self.max_embeddings_multiples = max_embeddings_multiples
        self.highres_fix_scale = highres_fix_scale
        self.highres_fix_save_1st = highres_fix_save_1st
        self.negative_scale = negative_scale

repo_dir = "stitch_diffusion/kohya-trainer"
network_weight = "stitch_diffusion/lora.safetensors"  # @param {'type':'string'}
network_mul = 0.7
network_module = "networks.lora"
network_args = ""

v2 = True
v_parameterization = False
prompt = "360-degree panoramic image, Hogwarts campus, hyper realistic"  # @param {type: "string"}
negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
model = "stitch_diffusion/pretrained_model/stable-diffusion-2-1-base.safetensors"  # @param {twgetype: "string"}
vae = "stitch_diffusion/vae/stablediffusion.vae.pt"  # @param {type: "string"}
outdir = "stitch_diffusion/output"  # @param {type: "string"}
scale = 7
sampler = "ddim"
steps = 50 # @param {type: "integer"}
precision = "fp16"
width = 2048
height = 512
images_per_prompt = 1
batch_size = 1  # @param {type: "integer"}
clip_skip = 2
seed = 11  # @param {type: "integer"}

final_prompt = f"{prompt} --n {negative}"

config = {
    "v2": v2,
    "v_parameterization": v_parameterization,
    "network_module": network_module,
    "network_weight": network_weight,
    "network_mul": float(network_mul),
    "network_args": eval(network_args) if network_args else None,
    "ckpt": model,
    "outdir": outdir,
    "xformers": True,
    "vae": vae if vae else None,
    "fp16": True,
    "W": width,
    "H": height,
    "seed": seed if seed > 0 else None,
    "scale": scale,
    "sampler": sampler,
    "steps": steps,
    "max_embeddings_multiples": 3,
    "batch_size": batch_size,
    "images_per_prompt": images_per_prompt,
    "clip_skip": clip_skip if not v2 else None,
    "prompt": final_prompt,
}
my_args = StitchDiffusionArgs(
    v2=config['v2'],
    v_parameterization=config['v_parameterization'],
    network_module=[config['network_module']],
    network_weights=[config['network_weight']],
    network_mul=[config['network_mul']],
    network_args=eval(config['network_args']) if config['network_args'] else None,
    ckpt=config['ckpt'],
    outdir=config['outdir'],
    xformers=config['xformers'],
    vae=config['vae'] if config['vae'] else None,
    fp16=config['fp16'],
    W=config['W'],
    H=config['H'],
    seed=config['seed'] if config['seed'] > 0 else None,
    scale=config['scale'],
    sampler=config['sampler'],
    steps=config['steps'],
    max_embeddings_multiples=config['max_embeddings_multiples'],
    batch_size=config['batch_size'],
    images_per_prompt=config['images_per_prompt'],
    clip_skip=config['clip_skip'] if not config['v2'] else None,
    prompt=config['prompt']
)
negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

class StitchDiffusionPipeline():
    def __init__(self, args: StitchDiffusionArgs):
        self.args = args
    def __call__(self, prompt):
        self.args.prompt = f"360-degree panoramic image, {prompt} --n {negative}"
        image = main(self.args)
        return image

class StitchDiffusion():
    def __init__(self, args: StitchDiffusionArgs):
        self.pipe = StitchDiffusionPipeline (args)
    def inference(self,prompt,savename):
        image = self.pipe(prompt)[0]
        image.save(savename)

if __name__ == "__main__":
    sd = StitchDiffusion(my_args)
    sd.inference("360-degree panoramic image, basketball court on a mountain", "ye.png")