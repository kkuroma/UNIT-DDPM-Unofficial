import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import configs

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """ 
    Returns a linear schedule of betas from start to end with an input timestep
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device=device, noise=None):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    if noise is None:
      noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # formula ->
    # mean = square root of alpha prod
    # std = square root of 1 - alpha prod
    mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
    var = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    # mean + variance
    return  mean+var, noise.to(device)


# Define beta schedule
timesteps = configs.TIMESTEPS
betas = linear_beta_schedule(timesteps=timesteps)

# Pre-calculate different terms for closed form
# alpha = 1-beta
alphas = 1. - betas
# alphas_cumprod = prod(alpha)
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

distance_fn = nn.MSELoss()
LAMBDA = 10

def train_one_epoch(ep, models, optimizers, data_loaderA, data_loaderB, identity = False, iterations=100):
  # gen1 = generator for domain 1
  # gen2 = generator for domain 2
  # cyc12 = transform domain 1 to domain 2
  # cyc21 = transform domain 2 to domain 1
  genA, genB, cycAB, cycBA = models
  genA.train()
  genB.train()
  cycAB.train()
  cycBA.train()

  #define optimizers
  optim_gen, optim_cyc = optimizers

  #tqdm loader
  pbar = tqdm(range(iterations), desc = f'Training Epoch {ep}')

  for i in pbar:

    #randonly sample images from 2 domain
    optim_gen.zero_grad()
    optim_cyc.zero_grad()
    xA0 = next(iter(data_loaderA))[0].to(device)
    xB0 = next(iter(data_loaderB))[0].to(device)

    # begin diffusion
    tA = torch.randint(0, timesteps, (xA0.shape[0],), device=device).long()
    tB = torch.randint(0, timesteps, (xB0.shape[0],), device=device).long()
    noiseA = torch.randn_like(xA0, device=device)
    noiseB = torch.randn_like(xB0, device=device)
    xB0_fake = cycAB(xA0)
    xA0_fake = cycBA(xB0)

    #add noise
    xAtA, _ = forward_diffusion_sample(xA0, tA, device, noiseA)
    xBtB, _ = forward_diffusion_sample(xB0, tB, device, noiseB)
    xAtB_fake, _ = forward_diffusion_sample(xA0_fake, tB, device, noiseA)
    xBtA_fake, _ = forward_diffusion_sample(xB0_fake, tA, device, noiseB)

    #update diffusion weight
    predA = genA(torch.cat([xAtA, xBtA_fake.detach()], dim=1), tA)
    predB = genB(torch.cat([xBtB, xAtB_fake.detach()], dim=1), tB)
    diffusion_loss = distance_fn(predA,noiseA) + distance_fn(predB,noiseB)
    diffusion_loss.backward()
    optim_gen.step()

    #update cycle weight
    predA = genA(torch.cat([xAtA, xBtA_fake], dim=1), tA)
    predA_fake = genA(torch.cat([xAtB_fake, xBtB], dim=1), tB)
    predB = genB(torch.cat([xBtB, xAtB_fake], dim=1), tB)
    predB_fake = genB(torch.cat([xBtA_fake, xAtA], dim=1), tA)
    cyc_loss = distance_fn(predA,noiseA) + distance_fn(predB,noiseB) + distance_fn(predA_fake,noiseA) + distance_fn(predB_fake,noiseB)

    if identity:
      xA0_cyc = cycBA(xB0_fake)
      xB0_cyc = cycAB(xA0_fake)
      cyc_loss += distance_fn(xA0_cyc, xA0)*LAMBDA + distance_fn(xB0_cyc, xB0)*LAMBDA
    cyc_loss.backward()
    optim_cyc.step()

    pbar.set_description(f"Epoch {ep} - Step {i+1}/{iterations} - Diff Loss = {round(diffusion_loss.item(),4)} - Cyc Loss = {round(cyc_loss.item(),4)}")
    
@torch.no_grad()
def translate_before_release(xA0, xBt, t, model):
  '''
  removes noise from a noisy image at timestep t before release time
  unless we're sampling the final image - some noise is sampled to increase variance
  '''
  noiseA = torch.randn_like(xA0)
  noiseB = torch.randn_like(xBt)
  if t.min() <= e:
    noiseA = noiseA*0
    noiseB = noiseB*0
  xAt, _ = forward_diffusion_sample(xA0, t, device, noiseA)

  betas_t = get_index_from_list(betas, t, xBt.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, xBt.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, xBt.shape)
  predicted_noise = model(torch.cat([xBt, xAt], dim=1), t)
  model_mean = sqrt_recip_alphas_t * (xBt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, xBt.shape)
  xBt_minus1 = model_mean + torch.sqrt(posterior_variance_t) * noiseB
  return xBt_minus1

@torch.no_grad()
def translate_after_release(xAt, xBt, t, model):
  '''
  removes noise from a noisy image at timestep t before release time
  unless we're sampling the final image - some noise is sampled to increase variance
  '''
  noiseA = torch.randn_like(xAt)
  noiseB = torch.randn_like(xBt)
  if t.min() <= e:
    noiseA = noiseA*0
    noiseB = noiseB*0

  betas_t = get_index_from_list(betas, t, xBt.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, xBt.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, xBt.shape)
  predicted_noise = model(torch.cat([xBt, xAt], dim=1), t)
  model_mean = sqrt_recip_alphas_t * (xBt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, xBt.shape)
  xBt_minus1 = model_mean + torch.sqrt(posterior_variance_t) * noiseB

  betas_t = get_index_from_list(betas, t, xAt.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, xAt.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, xAt.shape)
  predicted_noise = model(torch.cat([xAt, xBt], dim=1), t)
  model_mean = sqrt_recip_alphas_t * (xAt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, xAt.shape)
  xAt_minus1 = model_mean + torch.sqrt(posterior_variance_t) * noiseA

  return xAt_minus1, xBt_minus1

@torch.no_grad()
def translate(xA0, model, release_time=1):
  model.eval()
  b = xA0.shape[0]
  xBt = torch.randn(xA0.shape).to(device)
  noiseA = torch.randn_like(xA0)
  t = torch.full((b,), max(release_time,0), device=device, dtype=torch.long)
  xAt, _ = forward_diffusion_sample(xA0, t, device, noiseA)
  for i in tqdm(range(timesteps)):
    time = timesteps-i-1
    t = torch.full((b,), time, device=device, dtype=torch.long)
    if time>release_time:
      xBt = translate_before_release(xA0, xBt, t, model)
    else:
      xAt, xBt = translate_after_release(xAt, xBt, t, model)
  return xBt