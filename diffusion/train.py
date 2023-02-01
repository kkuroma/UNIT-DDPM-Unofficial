from AttnUnet import Unet
from PlainUnet import SimpleUnet_plain
from torch.optim import Adam
import configs 
from loaders import anime_loader, human_loader
from diffusion import train_one_epoch

device = configs.DEVICE

def get_model(plain=False):
  if not plain:
    model = Unet(
        configs.DIM, 
        channels = 6,
        out_dim = 3,
        dim_mults = (1, 2, 4, 8, 8),
    )
  else:
    model = SimpleUnet_plain(
        in_dim=3, 
        dim=dim, 
        out_dim=3, 
    )
  print("Num params: ", sum(p.numel() for p in model.parameters()))
  return model
  
if __name__=="__main__":

    #model 1 - generate anime girls
    anime_gen = get_model().to(device)
    #model 2 - generate human face
    human_gen = get_model().to(device)
    #model 3 - anime -> human
    anime2human = get_model(plain=True).to(device)
    #model 4 - human -> anime
    human2anime = get_model(plain=True).to(device)

    #optimizers 
    optim_gen = Adam(list(anime_gen.parameters())+list(human_gen.parameters()), lr=configs.LR)
    optim_cyc = Adam(list(anime2human.parameters())+list(human2anime.parameters()), configs.LR)

    if configs.LOAD_FROM_CHECKPOINT:

      anime_gen.load_state_dict(torch.load(CHECKPOINT_DIR+'anime.pt'))
      human_gen.load_state_dict(torch.load(CHECKPOINT_DIR+'human.pt'))
      anime2human.load_state_dict(torch.load(CHECKPOINT_DIR+'anime2human.pt'))
      human2anime.load_state_dict(torch.load(CHECKPOINT_DIR+'human2anime.pt'))

    models = [
        anime_gen,
        human_gen,
        anime2human,
        human2anime
    ]

    optimizers = [
        optim_gen,
        optim_cyc
    ]

    for ep in range(1,configs.EPOCHS):
      train_one_epoch(ep, models, optimizers, anime_loader, human_loader, identity = True)
      torch.save(anime_gen.state_dict(), CHECKPOINT_DIR+'anime.pt')
      torch.save(human_gen.state_dict(), CHECKPOINT_DIR+'human.pt')
      torch.save(anime2human.state_dict(), CHECKPOINT_DIR+'anime2human.pt')
      torch.save(human2anime.state_dict(), CHECKPOINT_DIR+'human2anime.pt')
 