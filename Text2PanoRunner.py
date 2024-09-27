'''
Tejas Bharadwaj, UCLA. 3.1.24
class which runs the prompt engineering and generates a panoramic image using
idea2img and StitchDiffusion
'''
import os

class Text2PanoRunner:
    def __init__(self, api_key=None, testfile="testsample.txt", num_img=1, num_prompt=3, max_rounds=3,
                 verbose=False, foldername="candidates", strength=1.00, final_name='image'):
        self.config = {
            "api_key": api_key,
            "testfile": testfile,
            "num_img": num_img,
            "num_prompt": num_prompt,
            "max_rounds": max_rounds,
            "verbose": verbose,
            "foldername": foldername,
            "strength": strength,
            "final_name": final_name
        }
        self.args = ""
        for k, v in self.config.items():
            if k.startswith("_"):
                self.args += f'"{v}" '
            elif isinstance(v, str):
                self.args += f'--{k}="{v}" '
            elif isinstance(v, bool) and v:
                self.args += f"--{k} "
            elif isinstance(v, float) and not isinstance(v, bool):
                self.args += f"--{k}={v} "
            elif isinstance(v, int) and not isinstance(v, bool):
                self.args += f"--{k}={v} "
        
    def run_command(self, filename = 'text2pano_self_refine_pipeline.py'):
        final_args = f"python {filename} {self.args}"
        os.system(final_args)



