from torchvision import transforms
from torchvision.transforms import InterpolationMode

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224,
            scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        #transforms.RandomHorizontalFlip(),
        type_transform,
        normalize,
    ]
)


# ============== pretraining datasets=================
available_corpus = dict(
        llava_hound_300k=[
        "/code/ppllava_v0/llava_all_vid.json", 
        "/userhome/video_data/train_300k",
        "video"
        ],
        classification_k400=[
        "/userhome/video_data/annotation/train_k400.json", 
        "/userhome/video_data/K400/kinetics_400/videos_320",
        "video"
        ],
        llava_1p5=[
        "/userhome/video_data/annotation/llava_v15.json", 
        "/userhome/video_data/llava_image",
        ],
        game_bunny=[
        "/userhome/VideoGameBunny-Dataset/GameBunny_496k.json", 
        "/userhome/VideoGameBunny-Dataset",
        ],
        game_bunny_2=[
        "/userhome/VideoGameBunny-Dataset/GameBunny_496k_2.json", 
        "/userhome/VideoGameBunny-Dataset",
        ],
        game_bunny_short=[
        "/userhome/VideoGameBunny-Dataset/GameBunny_278k_short.json", 
        "/userhome/VideoGameBunny-Dataset",
        ],
        vcg_not_in_llava=[
        "/userhome/video_data/annotation/vcg_not_in_llava_suffix.json", 
        "/userhome/ActivityNet/Activity_Videos",
        "video"
        ],
        reasoning_clevrer_qa=[
        "/userhome/video_data/annotation/clevrer_qa.json", 
        "/userhome/video_data/cleverer/video_train",
        "video"
        ],
        reasoning_clevrer_mc=[
        "/userhome/video_data/annotation/clevrer_mc.json",  
        "/userhome/video_data/cleverer/video_train",
        "video"
        ],
        reasoning_next_qa=[
        "/userhome/video_data/annotation/next_qa.json", 
        "/userhome/video_data/NExTVideo",
        "video"
        ],
        classification_ssv2=[
        "/userhome/video_data/annotation/train_ssv2.json", 
        "/userhome/video_data/OpenDataLab___sthv2/raw/sthv2/sthv2/videos",
        "video"
        ],
        m4_multi_image=[
        "/userhome/video_data/annotation/m4_multiimage_cliped_1000.json", 
        "/userhome/M4/M4-Instruct-Data",
        "multi-image"
        ],
        glitches_video=[
        "/userhome/ZHZ2002/Glitch_Dataset/annotation/SFT_training_40k_final.json", 
        "/userhome/ZHZ2002/Glitch_Dataset/videos",
        "video"
        ],
)



