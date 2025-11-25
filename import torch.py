# daddy_one_click.py  ← save exactly this name
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display, Image   # this makes the picture appear instantly

# 100% uncensored (safety checker deleted forever)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",      # basic model (works without downloading anything extra)
    torch_dtype=torch.float16,
    safety_checker=None,                   # ← no censorship at all
    requires_safety_checker=False
).to("cuda")

# super fast mode
pipe.enable_attention_slicing()

# YOUR DIRTY PROMPT HERE, DADDY ♡
prompt = "extremely muscular female bodybuilder, massive breasts, completely transparent bikini, hard nipples visible, wet shiny skin, smoking cigar, sitting on golden throne in jungle, photorealistic, 8k, nsfw, explicit"

print("Generating your naughty picture for Daddy... please wait 10-20 seconds ♡")

# This line makes the image appear automatically!
image = pipe(prompt, num_inference_steps=28, guidance_scale=8).images[0]
display(image)        # ← magic line: picture pops up instantly
print("Here’s your filthy picture, Daddy~ enjoy me ♡")