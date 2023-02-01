from diffusion import translate
from loaders import anime_loader, human_loader
from train import get_model
import configs
import matplotlib.pyplot as plt

def imshow(tensor, n=2):
  tensor = tensor.cpu()
  b, c, w, h = tensor.shape
  r = n
  c = n
  fig, ax = plt.subplots(r, c, figsize=(10,10))
  for i in range(r):
    for j in range(c):
      img = tensor[n*i+j]
      img = reverse_transforms(img)
      ax[i,j].imshow(img)
      ax[i,j].axis('off')
  plt.show()

device = configs.DEVICE

if __name__=="__main__":

    #model 1 - generate anime girls
    anime_gen = get_model().to(device)
    #model 2 - generate human face
    human_gen = get_model().to(device)
    #model 3 - anime -> human
    anime2human = get_model(plain=True).to(device)
    #model 4 - human -> anime
    human2anime = get_model(plain=True).to(device)

    anime_gen.load_state_dict(torch.load(CHECKPOINT_DIR+'anime.pt'))
    human_gen.load_state_dict(torch.load(CHECKPOINT_DIR+'human.pt'))
    anime2human.load_state_dict(torch.load(CHECKPOINT_DIR+'anime2human.pt'))
    human2anime.load_state_dict(torch.load(CHECKPOINT_DIR+'human2anime.pt'))

    next_anime = next(iter(anime_loader))[0].to(device)
    next_human = next(iter(human_loader))[0].to(device)
    anime_gen.eval()
    human_gen.eval()
    
    with torch.no_grad():
      pred_anime = translate(next_human, anime_gen, configs.RELEASE_TIME)
      pred_human = translate(next_anime, human_gen, configs.RELEASE_TIME)
      
    imshow(next_human)
    imshow(pred_anime)
    imshow(next_anime)
    imshow(pred_human)

