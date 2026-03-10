import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import top_k_top_p_filtering


rescale = lambda x: (x + 1.) / 2.



def save_img(xstart, fname):
    """Save a BATCH of images (B, H, W, C) where values are [0, 1]"""
    if len(xstart.shape) == 4:
        I = (xstart[0].clip(0,1)*255).astype(np.uint8)
    else:
        I = (xstart.clip(0,1)*255).astype(np.uint8)
    Image.fromarray(I).save(fname)

def torch_bchw_to_np_bhwc(x):
    """Converts torch tensor (B, C, H, W) in [-1, 1] to numpy (B, H, W, C) in [0, 1]"""
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def pad_to_M(x, M):
    hp = math.ceil(x.shape[2]/M)*M-x.shape[2]
    wp = math.ceil(x.shape[3]/M)*M-x.shape[3]
    x = torch.nn.functional.pad(x, (0,wp,0,hp,0,0,0,0))
    return x

@torch.no_grad()
def run_non_interactive(model, dsets, index, output_path, temperature=1.0, top_k=1200, half_sample=False, scale_factor=1.0): # top_k = 100 initial
    top_k = 50
    top_p = 1.0
    temperature = 1.0
    """
    Runs the conditional sampling logic without Streamlit interaction.
    """
    print(f"--- Running non-interactive generation (Index: {index}) ---")
    
    if len(dsets.datasets) > 1:
        dset = next(iter(dsets.datasets.values())) 
    else:
        dset = next(iter(dsets.datasets.values()))
        
    batch_size = 8
    if len(dset) / batch_size * batch_size != len(dset):
        raise Error('batchsize is not divided by dataset')
    if index >= len(dset) or index < 0:
        raise IndexError(f"Index {index} out of range for dataset size {len(dset)}")
    # print(len(dset))
    # exit(0)
    # indices = list(range(0, len(dset))) # list(range(index, index+batch_size))
     
    m_compare = os.makedirs(os.path.join(output_path, 'compare'), exist_ok=True)
    m_rec = os.makedirs(os.path.join(output_path, 'rec'), exist_ok=True)
    m_condition = os.makedirs(os.path.join(output_path, 'condition'), exist_ok=True)
    m_start = os.makedirs(os.path.join(output_path, 'start'), exist_ok=True)
    m_gen = os.makedirs(os.path.join(output_path, 'gen'), exist_ok=True)
    
    for indice in range(len(dset)//batch_size):
        example = default_collate([dset[i] for i in list(range(indice*batch_size, (indice+1)*batch_size))]) # default_collate([dset[indice]]) # default_collate([dset[i] for i in indices])
    
        x = model.get_input("image", example).to(model.device)
        cond_key = model.cond_stage_key
        c = model.get_input(cond_key, example).to(model.device)
    
        if scale_factor != 1.0:
            x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="bicubic")
            c = torch.nn.functional.interpolate(c, scale_factor=scale_factor, mode="bicubic")
    
        quant_z, z_indices = model.encode_to_z(x)

        quant_c, c_indices = model.encode_to_c(c)
        cshape = quant_z.shape
        
        xrec = model.first_stage_model.decode(quant_z)
        
        x_np = torch_bchw_to_np_bhwc(x)
        xrec_np = torch_bchw_to_np_bhwc(xrec)
        # print(x_np.shape, xrec_np.shape)
        # exit(0)
        for subindex in range(batch_size):
            # print(x_np[subindex,:,:,:].shape)
            # print(x_np[subindex].shape)
            save_img(x_np[subindex,:,:,:], os.path.join(output_path, "compare", f"{indice*batch_size+subindex:06d}.png"))
            save_img(xrec_np[subindex,:,:,:], os.path.join(output_path, "rec", f"{indice*batch_size+subindex:06d}.png"))
            print(f"Saved input and reconstruction to {output_path}")
    
        if cond_key == "segmentation":
            num_classes = c.shape[1]
            c_argmax = torch.argmax(c, dim=1, keepdim=True)
            c_onehot = torch.nn.functional.one_hot(c_argmax, num_classes=num_classes)
            c_rgb = c_onehot.squeeze(1).permute(0, 3, 1, 2).float()
            c_rgb = model.cond_stage_model.to_rgb(c_rgb) 
        else:
            c_rgb = c 
    
        c_np = rescale(c_rgb.detach().cpu().permute(0,2,3,1).view(batch_size, 256, 256).numpy()) # torch_bchw_to_np_bhwc(c_rgb)
        # print(c_np.shape)
        # print(c_rgb.min(), c_rgb.max())
        # print(c_np.min(), c_np.max())
        # exit(0)
        for subindex in range(batch_size):
            save_img(c_np[subindex, :, :], os.path.join(output_path, "condition", f"{indice*batch_size+subindex:06d}.png"))
            print(f"Saved condition ({cond_key}) to {output_path}/{indice*batch_size+subindex}.png")
    
    
        idx = z_indices
    
        if half_sample:
            print("Running in Image Completion mode.")
            start = idx.shape[1]//2
        else:
            start = 0
    
        idx[:,start:] = 0
        idx = idx.reshape(cshape[0],cshape[2],cshape[3])
        start_i = start//cshape[3]
        start_j = start %cshape[3]
    
        if not half_sample and quant_z.shape == quant_c.shape:
            print("Setting starting indices to c_indices (Conditional Generation mode).")
            idx = c_indices.clone().reshape(cshape[0],cshape[2],cshape[3])
    
        cidx = c_indices
        cidx = cidx.reshape(quant_c.shape[0],quant_c.shape[2],quant_c.shape[3])
    
        xstart = model.decode_to_img(idx[:,:cshape[2],:cshape[3]], cshape)
        # save_img(torch_bchw_to_np_bhwc(xstart), os.path.join(output_path, "start", f"{indice:06d}.png"))
    
        print(f"Starting sampling loop at ({start_i}, {start_j}) with T={temperature}, TopK={top_k}")
        
        start_t = time.time()
        total_steps = cshape[2] * cshape[3]
        
        for i in range(start_i,cshape[2]):
            for j in range(start_j,cshape[3]):
                
                if i <= 8:
                    local_i = i
                elif cshape[2]-i < 8:
                    local_i = 16-(cshape[2]-i)
                else:
                    local_i = 8
                
                if j <= 8:
                    local_j = j
                elif cshape[3]-j < 8:
                    local_j = 16-(cshape[3]-j)
                else:
                    local_j = 8
    
                i_start = i-local_i
                i_end = i_start+16
                j_start = j-local_j
                j_end = j_start+16
                
                patch = idx[:,i_start:i_end,j_start:j_end]
                patch = patch.reshape(patch.shape[0],-1)
                cpatch = cidx[:, i_start:i_end, j_start:j_end]
                cpatch = cpatch.reshape(cpatch.shape[0], -1)
                patch = torch.cat((cpatch, patch), dim=1)
                
                logits,_ = model.transformer(patch[:,:-1])
                logits = logits[:, -256:, :]
                logits = logits.reshape(cshape[0],16,16,-1)
                logits = logits[:,local_i,local_j,:]
                
                logits = logits/temperature
                
                if top_k is not None:
                    logits = model.top_k_logits(logits, top_k)
                    # logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
               
                ix = torch.multinomial(probs, num_samples=1)
                idx[:,i,j] = ix.squeeze(-1)
                
                current_step = i * cshape[3] + j
                if current_step % 100 == 0:
                     elapsed = time.time() - start_t
                     print(f"Step {current_step}/{total_steps} | Time: {elapsed:.2f}s | Loc: ({i},{j})")
                     
        xstart = model.decode_to_img(idx[:,:cshape[2],:cshape[3]], cshape)
        xstart_np = torch_bchw_to_np_bhwc(xstart)
        for subindex in range(batch_size):
            final_output_file = os.path.join(output_path, "gen", f"{indice*batch_size+subindex:06d}.png")
            save_img(xstart_np[subindex,:,:,:], final_output_file)
            print(f"--- Finished sampling. Final image saved to {final_output_file} ---")


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to sample from. Corresponds to Streamlit's 'Example Index'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples",
        help="Directory to save the generated images."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sampling top-k."
    )
    parser.add_argument(
        "--half_sample",
        action="store_true",
        help="Enable Image Completion mode (samples the second half of the image)."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="last.ckpt",
        help="ckpt name"
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Warning: Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Warning: Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Warning: Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Warning: Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)    # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    
    config_model = OmegaConf.to_container(config.model, resolve=True)
    config_model = OmegaConf.create(config_model)

    model = load_model_from_config(config_model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    
    opt, unknown = parser.parse_known_args()

   
    ckpt = None
   
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"Logdir: {logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

 
    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu=True, eval_mode=True)
    print(f"Model loaded. Global step: {global_step}")
    
  
    os.makedirs(opt.output_dir, exist_ok=True)
    print(f"Output directory created: {opt.output_dir}")


    run_non_interactive(
        model, 
        dsets, 
        index=opt.index, 
        output_path=opt.output_dir, 
        temperature=opt.temp, 
        top_k=opt.top_k, 
        half_sample=opt.half_sample
    )