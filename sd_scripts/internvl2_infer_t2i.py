import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import requests
import io
import math
import re
import time

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
from transformers import AltCLIPModel, AutoProcessor
from safetensors import safe_open
import os
import argparse
from collections import defaultdict
import torch.distributed as dist

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# from mogao.models import MODEL_REGISTRY

# from byted_seed_preview.uploader import ImageUploader
import time
import sys
from datetime import timedelta

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    _, orig_height, orig_width = image.shape
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = F.resize(image, (target_height, target_width), interpolation=F.InterpolationMode.BICUBIC)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img[:, box[1]:box[3], box[0]:box[2]]
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = F.resize(image, (image_size, image_size), interpolation=F.InterpolationMode.BICUBIC)
        processed_images.append(thumbnail_img)
    return processed_images


# @MODEL_REGISTRY.register()
class Internvl2_instruct_dual(nn.Module):
    def __init__(self, device="cuda", bf16=True, r_size=512, vit_res=448, max_patch_num=12, tag_idxs='1,2', mark_4b=False, 
        cot=False, init_config='/mnt/bn/aigc-t2i/gaoyu/hf_cache/InternVL2-2B', **wargs):
        print(f"reward model res: {r_size}, vit_res: {vit_res}")
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.bf16 = bf16

        self.r_size = r_size
        self.vit_res = vit_res
        self.max_patch_num = max_patch_num
        path = init_config

        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.mark_4b = mark_4b
        if self.mark_4b:
            self.prompt_template = '<|system|>\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|end|><|user|>\n<user_prompt><|end|><|assistant|>\n'
        else:
            self.prompt_template = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n<user_prompt><|im_end|><|im_start|>assistant\n'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.en_yes_token_id = self.tokenizer('yes', return_attention_mask=False, add_special_tokens=False)['input_ids'][-1]
        self.en_no_token_id = self.tokenizer('no', return_attention_mask=False, add_special_tokens=False)['input_ids'][-1]
        self.cn_yes_token_id = self.tokenizer('是', return_attention_mask=False, add_special_tokens=False)['input_ids'][-1]
        self.cn_no_token_id = self.tokenizer('否', return_attention_mask=False, add_special_tokens=False)['input_ids'][-1]
        self.cn_prefix = self.tokenizer('是', return_attention_mask=False, add_special_tokens=False)['input_ids'][0]
        
        mg_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.cn_tag2condition = {
            1: '图文匹配，',
            2: '美感和图文匹配，',
            3: '视觉效果，',
            4: '结构，',
            5: '色彩，',
            6: '构图，'
        }
        self.en_tag2condition = {
            1: ' image-text matching,',
            2: ' aesthetic,',
            3: ' visual effect,',
            4: ' structure,',
            5: ' colour,',
            6: ' picture composition,'
        }
        self.tag_idxs = str(tag_idxs).split(',')
        self.cot = cot
        if self.cot:
            import copy
            self.model_ori = copy.deepcopy(self.model)
            self.model_ori.eval()

        if isinstance(vit_res, int):
            vit_res = (vit_res, vit_res)
        self.image_transform = Compose(
            [
                Resize(vit_res, interpolation=BICUBIC),
                Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.num_image_token = self.model.num_image_token
        downsample_ratio = 448//vit_res[0]
        self.num_image_token  = int(self.num_image_token / (downsample_ratio*downsample_ratio))

    def forward(self, batch_data, mode=None):
        pass

    def score_grad(self, prompt, cur_image, ref_image, do_ortho_list=[False], Parallel_weight_list=[1.0], Vertical_weight_list=[1.0], \
                   project_on_vision_list=[False], V_prompt_emb_list=[None], V_dynamic=None):
        def _has_chinese(input_str):
            return any("\u4e00" <= char <= "\u9fff" for char in input_str)
        
        def _is_last_char_punctuation(text):
            pattern = r'[,.!?;:\'"()（）\[\]{}，。！？；：、…《》【】\'"`"""]$'
            
            if not text:  # 如果字符串为空
                return False
                
            return bool(re.search(pattern, text))

        ## pixel_values
        input_size = self.vit_res
        max_num = self.max_patch_num
        if self.r_size is not None:
            cur_image = F.resize(cur_image, (self.r_size, self.r_size), interpolation=F.InterpolationMode.BICUBIC)
            ref_image = F.resize(ref_image, (self.r_size, self.r_size), interpolation=F.InterpolationMode.BICUBIC)
        image_list = [cur_image, ref_image]
        pixel_values_images = []
        for image in image_list:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [self.image_transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            pixel_values_images.append(pixel_values)
        pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model.dtype).cuda()

        if self.cot:
            cur_pixel_values_images = []
            cur_images = dynamic_preprocess(cur_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            cur_pixel_values = [self.image_transform(image) for image in cur_images]
            cur_pixel_values = torch.stack(cur_pixel_values)
            cur_pixel_values_images.append(cur_pixel_values)
            cur_pixel_values = torch.cat(cur_pixel_values_images, dim=0).to(self.model_ori.dtype).cuda()

        chinese_mark = False
        if _has_chinese(prompt):
            chinese_mark = True
            if not _is_last_char_punctuation(prompt):
                prompt = f'{prompt}。'
            prompt = f'\“{prompt}\”'
            cat_tag = ''
            for tag_idx in self.tag_idxs:
                tag = self.cn_tag2condition[int(tag_idx)]
                cat_tag+=tag
            cat_tag = cat_tag[:-1]

            if self.cot:
                first_round_instruct = "<image>\n请为这张图像生成对应的图像描述，字数在50词左右。"
                instruct = "<image>\n<image>\n这两幅图都是用这个文本生成的：<prompt>, 第一张图像描述的内容为 <caption_cur>, 请以这个描述作为参考。你觉得第一张图是否比第二张图好？请回答是还是否，评判标准包括：<tags>。"
            else:
                instruct = "<image>\n<image>\n这两幅图都是用这个文本生成的：<prompt> 你觉得第一张图是否比第二张图好？请回答是还是否，评判标准包括：<tags>。"
        else:
            if not _is_last_char_punctuation(prompt):
                prompt = f'{prompt}.'
            prompt = f'\"{prompt}\"'
            cat_tag = ''
            for tag_idx in self.tag_idxs:
                tag = self.en_tag2condition[int(tag_idx)]
                cat_tag+=tag
            cat_tag = cat_tag[:-1]

            if self.cot:
                first_round_instruct = "<image>\nPlease generate a corresponding image description for this image, with about 50 words."
                instruct = "<image>\n<image>\nboth of these images were generated using this text: <prompt> The first image description is <caption_cur>, please use this description as a reference. Do you think the first image is better than the second one? Please answer with 'yes' or 'no'. The evaluation criteria include: <tags>."
            else:
                instruct = "<image>\n<image>\nboth of these images were generated using this text: <prompt> Do you think the first image is better than the second one? Please answer with 'yes' or 'no'. The evaluation criteria include: <tags>."

        instruct = instruct.replace('<tags>', cat_tag)
        prompt_w_instruct = instruct.replace('<prompt>', prompt)

        if self.cot:
            # first round to get image description
            generation_config = dict(max_new_tokens=100, do_sample=True)
            with torch.no_grad():
                predicted_first_round = self.model_ori.chat(self.tokenizer, cur_pixel_values, first_round_instruct, generation_config, history=None, return_history=None)
            prompt_w_instruct = prompt_w_instruct.replace('<caption_cur>', predicted_first_round)

        num_patches_list = [1] * pixel_values.shape[0]
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        cur_prompt = self.prompt_template.replace('<user_prompt>', prompt_w_instruct)
        for num_patches in num_patches_list:
            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + self.IMG_END_TOKEN
            cur_prompt = cur_prompt.replace('<image>', image_tokens, 1)
        model_inputs = self.tokenizer(cur_prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        if chinese_mark and self.mark_4b:
            input_ids = torch.cat((input_ids, torch.tensor([[self.cn_prefix]]).to(input_ids.device)), dim=1)
            attention_mask = torch.cat((attention_mask, torch.tensor([[1]]).to(attention_mask.device)), dim=1)

        vit_embeds = self.model.extract_feature(pixel_values)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)

        try:
            attention_mask = attention_mask.to(self.device).to(self.model.language_model.output.weight.dtype)
            input_embeds = input_embeds.to(self.device).to(self.model.language_model.output.weight.dtype)
        except:
            attention_mask = attention_mask.to(self.device).to(self.model.language_model.lm_head.weight.dtype)
            input_embeds = input_embeds.to(self.device).to(self.model.language_model.lm_head.weight.dtype)
            
        inputs = {
            'attention_mask': attention_mask,
            'inputs_embeds': input_embeds
        }
        output = self.model.language_model(**inputs)
        last_logit = output[0][0][-1]

        if chinese_mark:
            yes_score = last_logit[self.cn_yes_token_id]
            no_score = last_logit[self.cn_no_token_id]
        else:
            yes_score = last_logit[self.en_yes_token_id]
            no_score = last_logit[self.en_no_token_id]
        reward = torch.exp(yes_score)/(torch.exp(yes_score)+torch.exp(no_score))
        return [reward], [yes_score], [no_score]


if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_config" , type= str , default= '/mnt/bn/aigc-t2i/gaoyu/hf_cache/InternVL2-26B')
    parser.add_argument("--output_dir" , type= str , default= '/mnt/bn/seed-aigc-aesthetic-lq/lifanshi/reward_models/for_fanshi/bash_infer/output'   )
    parser.add_argument("--model_path" , type= str , default= '/mnt/bn/seed-aigc-aesthetic-lq/lifanshi/reward_models/for_fanshi/intern_dual_vlm113.bin'    )
    parser.add_argument("--img_dir" , type= str)
    # parser.add_argument("--img_dir1" , type= str)
    # parser.add_argument("--img_dir2" , type= str)
    parser.add_argument("--prompt_path" , type= str)
    parser.add_argument("--json_input" , type= str, help="JSON file with list of {prompt, img1_path, img2_path, ...} items")
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=720))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    

    # 清理未使用的显存
    torch.cuda.empty_cache()
    # 重置显存统计信息
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_device(local_rank)
    
    # img_dir1 = args.img_dir1
    # img_dir2 = args.img_dir2
    img_dir = args.img_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path =  args.model_path
    prompt_path = args.prompt_path
    output_txt_path = os.path.join(output_dir, f"{rank:03}.json")
    
    mark_4b =  False
    init_config = args.init_config
    internvl2 = Internvl2_instruct_dual(device=torch.device(f'cuda:{local_rank}'), bf16=False, mark_4b=mark_4b,
        init_config=init_config, tag_idxs='1')
    loaded_state_dict = torch.load(model_path, map_location="cpu", mmap=True)
    internvl2.load_state_dict(loaded_state_dict, strict=True)
    internvl2.to(f'cuda:{local_rank}')
    internvl2.eval()

    import glob
    import cv2
    from collections import defaultdict
    import json

    def load_image_tensor(image_path):
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        r_img = np.transpose(img, (2, 0, 1))
        r_tensor = torch.tensor(r_img.astype(np.float32)) / 255.
        return r_tensor.to('cuda')

    start_time = time.time()

    if args.json_input:
        # ==================== JSON input mode ====================
        # Input: JSON file with a list of {prompt, img1_path, img2_path, ...}
        # Each item is an independent pairwise comparison.
        with open(args.json_input, 'r') as f:
            all_items = json.load(f)

        # 分布式分片
        total = len(all_items)
        each_split = total // world_size
        start_idx = rank * each_split
        end_idx = total if rank == world_size - 1 else (rank + 1) * each_split
        select_items = all_items[start_idx:end_idx]

        print(f'world_size {world_size}, rank {rank}, local_rank {local_rank}, '
              f'processing items [{start_idx}, {end_idx}) / {total}')

        output_json_path = os.path.join(output_dir, f"{rank:03}.json")
        w = open(output_json_path, 'w')
        print('start')

        for item_idx, item in enumerate(select_items):
            try:
                prompt = item['prompt']
                img1_path = item['img1_path']
                img2_path = item['img2_path']

                cur_img = load_image_tensor(img1_path)
                ref_img = load_image_tensor(img2_path)

                with torch.no_grad():
                    score_list = internvl2.score_grad(prompt, cur_img, ref_img)
                    score = score_list[0][0].item()
                    yes_s = score_list[1][0].item()
                    no_s = score_list[2][0].item()

                result = dict(item)
                result['reward_score'] = score
                result['yes_score'] = yes_s
                result['no_score'] = no_s
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

                print(f'[{item_idx}/{len(select_items)}] {img1_path} vs {img2_path}: '
                      f'reward={score:.4f}, yes={yes_s:.4f}, no={no_s:.4f}')

            except Exception as e:
                print(f"Error processing item {start_idx + item_idx}: {e}")
                continue

        w.close()
        print(f'{output_json_path} complete')

    else:
        # ==================== Legacy mode (prompt_path + img_dir) ====================
        complete_ids = set([])
        prompt_path = args.prompt_path
        img_dir = args.img_dir

        id2prompt_map = {}
        with open(prompt_path) as lines:
            for line in lines:
                row = json.loads(line)
                prompt = row['inputs'][0]['txt']
                idx = int(row['prompt_index'])
                id2prompt_map[idx] = prompt

        img_paths = []
        if img_dir is not None:
            img_paths += glob.glob(f'{img_dir}/*')

        img_scores = {}
        id2image_map = defaultdict(list)
        img_scores_float = {}
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            idx = int(img_name.split('_')[0])
            if idx in complete_ids:
                continue
            id2image_map[idx].append(img_path)
            img_scores[img_path] = 0.0
            img_scores_float[img_path] = 0.0

        idxs = list(id2image_map.keys())
        idxs.sort()
        id_nums = len(id2image_map)
        each_split_nums = id_nums // world_size
        cur_num = rank + 1
        split_num = world_size
        if cur_num != split_num:
            select_idxs = idxs[(cur_num-1)*each_split_nums: cur_num*each_split_nums].copy()
        else:
            select_idxs = idxs[(cur_num-1)*each_split_nums:].copy()
        print('select_idxs:', select_idxs)

        print('world_size', world_size, 'whole_rank', rank, 'rank_num:', local_rank,
              'select_idxs num:', len(select_idxs), 'start_id', select_idxs[0])

        output_txt_path = os.path.join(output_dir, f"{rank:03}.json")
        w = open(output_txt_path, 'w')
        print('start')

        for idx in select_idxs:
            image_paths = id2image_map[idx]
            try:
                prompt = id2prompt_map[idx]
                img_dict = {}
                total_scores = []
                for i in range(0, len(image_paths)):
                    image_path1 = image_paths[i]
                    img_dict[i] = (load_image_tensor(image_path1), image_path1)

                for i in range(0, len(image_paths)):
                    for j in range(i+1, len(image_paths)):
                        cur_img, image_path1 = img_dict[i]
                        ref_img, image_path2 = img_dict[j]

                        with torch.no_grad():
                            score_list = internvl2.score_grad(prompt, cur_img, ref_img)
                            score = score_list[0][0].item()
                            yes_s = score_list[1][0].item()
                            no_s = score_list[2][0].item()
                            total_scores.append(score)

                        line = f'{image_path1} vs {image_path2}: {score}, yes: {yes_s}, no: {no_s}'
                        print(line)

                        if score > 0.5:
                            img_scores[image_path1] += 1.0
                            img_scores_float[image_path1] += score
                        elif score < 0.5:
                            img_scores[image_path2] += 1.0
                            img_scores_float[image_path2] += (1.0 - score)

                variance_population = np.var(total_scores)
                sort_list = []
                for image_path in image_paths:
                    sort_list.append((image_path, img_scores[image_path], img_scores_float[image_path]/3))
                sort_list.sort(key=lambda x: x[1], reverse=True)
                line = ''
                for image_path, score, float_score in sort_list:
                    line += f'{image_path},{score},{float_score}\t'
                line = line[:-1] + '\t' + prompt.replace("\n", "<sep_n>").replace("\t", "<sep_t>") + '\t' + str(variance_population) + '\n'
                w.write(line)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue

        w.close()
        print(f'{output_txt_path} complete')

    end_time = time.time()
    print(f"代码执行耗时: {end_time - start_time} 秒")


# CUDA_VISIBLE_DEVICES=7 python3 mogao/models/reward_models/Internvl2_instruct_dual_batch_infer_complete.py /mnt/bn/seed-aigc-hl/gaoyu/data/vis_data2/tuwen_8_32 /mnt/bn/seed-aigc-hl/gaoyu/model/intern_dual_tuwen_pretrain_vlm29.bin 0 /mnt/bn/seed-aigc-hl/wujie/reward_model/InternVL2-8B 8 32
# CUDA_VISIBLE_DEVICES=0 python3 mogao/models/reward_models/Internvl2_instruct_dual_batch_infer_complete.py /mnt/bn/seed-aigc-hl/gaoyu/data/vis_data2/tuwen_1_32 /mnt/bn/seed-aigc-hl/gaoyu/model/intern_dual_tuwen_pretrain_vlm29.bin 0 /mnt/bn/seed-aigc-hl/wujie/reward_model/InternVL2-8B

# CUDA_VISIBLE_DEVICES=7 python3 mogao/models/reward_models/Internvl2_instruct_dual_batch_infer_complete.py /mnt/bn/seed-aigc-hl/gaoyu/data/mingan_tag_0212_over_mix_sft_second/tuwen_16_16 /mnt/bn/seed-aigc-hl/gaoyu/model/intern_dual_tuwen_pretrain_vlm29.bin 0 /mnt/bn/seed-aigc-hl/wujie/reward_model