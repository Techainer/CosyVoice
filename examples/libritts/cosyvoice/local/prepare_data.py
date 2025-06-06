import argparse
import os
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import soundfile

from file import list_files

def process_filelist(filelist, output_dir):
    """
    Processes the filelist to create wav.scp, text, utt2spk, and spk2utt files.
    """
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}

    for line in tqdm(filelist):
        wav, spk, text = line.split('|')
        utt = os.path.basename(wav).replace('.wav', '')
        utt2wav[utt] = wav
        utt2text[utt] = text
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/wav.scp', 'w', encoding='utf-8') as f:
        for utt, wav in utt2wav.items():
            f.write(f'{utt} {wav}\n')

    with open(f'{output_dir}/text', 'w', encoding='utf-8') as f:
        for utt, text in utt2text.items():
            f.write(f'{utt} {text}\n')

    with open(f'{output_dir}/utt2spk', 'w', encoding='utf-8') as f:
        for utt, spk in utt2spk.items():
            f.write(f'{utt} {spk}\n')

    with open(f'{output_dir}/spk2utt', 'w', encoding='utf-8') as f:
        for spk, utts in spk2utt.items():
            f.write(f'{spk} {" ".join(utts)}\n')

    logger.success(f'Created utt2wav, utt2text, utt2spk, and spk2utt files in {output_dir}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--valid_size", type=float, default=0.05, help="Proportion of validation set size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'Loading metadata from {args.data_dir}')
    files = list_files(args.data_dir, extensions=[".txt"], recursive=True)
    
    train_filelist = []
    valid_filelist = []
    
    total_duration = 0

    for file in files:
        # if 'data_bac_nam' not in str(file):
        #     continue

        not_exist_count = 0
        subset_filelist = []

        if file.name == "metadata_lower.txt":
            logger.info(f'Loading {file}')
            spk = os.path.dirname(file).split("/")[-1]
            with open(file, "r", encoding="utf-8") as f:
                not_exist_count = 0
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split("|")
                    # if len(line) != 3:
                    #     print(line)
                    #     continue
                    filepath = line[0]
                    # filepath = filepath.replace('/home/andrew/data/tts', '/data/tts')
                    if not os.path.exists(filepath):
                        print(f"File {filepath} not exist")
                        not_exist_count += 1
                        continue
                    try:
                        y, sr = soundfile.read(filepath)
                    except:
                        print(f"File {filepath} cannot be read")
                        not_exist_count += 1
                        continue
                    if len(y.shape) != 1:
                        print(f"File {filepath} is not mono")
                        not_exist_count += 1
                        continue
                    if y.shape[0]/ sr > 20:
                        print(f"File {filepath} is too long")
                        not_exist_count += 1
                        continue
                    if y.shape[0]/ sr < 0.5:
                        print(f"File {filepath} is too short")
                        not_exist_count += 1
                        continue
                    if 'vivos' in str(file):
                        line[1] = line[1].lower().strip()
                    text = line[1]
                    if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
                        text += "."
                    text = text.replace(" .", ".").replace(" ,", ",")
                    text = " ".join(text.split()).lower()
                    line = f"{filepath}|{spk}|{text}"
                    subset_filelist.append(line)
                    total_duration += y.shape[0]/ sr

        elif file.name == "transcripts_lower.txt" and "vivos" not in str(file):
            logger.info(f'Loading {file}')
            spk = os.path.dirname(file).split("/")[4]
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split("|")
                    filepath = line[0]
                    # filepath = filepath.replace('/home/andrew/data/tts', '/data/tts')
                    if not os.path.exists(filepath):
                        print(f"File {filepath} not exist")
                        not_exist_count += 1
                        continue
                    try:
                        y, sr = soundfile.read(filepath)
                    except:
                        print(f"File {filepath} cannot be read")
                        not_exist_count += 1
                        continue
                    if len(y.shape) != 1:
                        print(f"File {filepath} is not mono")
                        not_exist_count += 1
                        continue
                    if y.shape[0]/ sr > 20:
                        print(f"File {filepath} is too long")
                        not_exist_count += 1
                        continue
                    if y.shape[0]/ sr < 1:
                        print(f"File {filepath} is too short")
                        not_exist_count += 1
                        continue
                    if 'vivos' in str(file):
                        line[1] = line[1].lower().strip()
                    text = line[1]
                    if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
                        text += "."
                    text = text.replace(" .", ".").replace(" ,", ",")
                    text = " ".join(text.split()).lower()
                    line = f"{filepath}|{spk}|{text}"
                    subset_filelist.append(line)
                    total_duration += y.shape[0]/ sr

        else:
            continue

        if not_exist_count > 0:
            logger.info(f"Not valid count: {not_exist_count}/{len(lines)}")  

        # Split the current subset into train and valid sets
        train_subset, valid_subset = train_test_split(subset_filelist, test_size=args.valid_size, random_state=42)
        train_filelist.extend(train_subset)
        valid_filelist.extend(valid_subset)
    
    logger.info(f"Total duration: {total_duration/3600:.2f} hours")

    # Save the training and validation file lists
    with open(os.path.join(args.output_dir, "train_filelist.txt"), "w", encoding="utf-8") as f:
        for file in train_filelist:
            f.write(file + "\n")

    with open(os.path.join(args.output_dir, "valid_filelist.txt"), "w", encoding="utf-8") as f:
        for file in valid_filelist:
            f.write(file + "\n")

    # Process training and validation filelists to create utt2wav, utt2text, utt2spk, and spk2utt files
    logger.info(f'Processing train filelist {len(train_filelist)} samples...')
    process_filelist(train_filelist, os.path.join(args.output_dir, "train"))

    logger.info(f'Processing valid filelist {len(valid_filelist)} samples...')
    process_filelist(valid_filelist, os.path.join(args.output_dir, "valid"))

    logger.success(f'Finished processing training and validation datasets. Saved to {args.output_dir}')

if __name__ == "__main__":
    main()