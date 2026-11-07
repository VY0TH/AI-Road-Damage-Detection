import json

nb_path = 'notebooks/Road_Damage_Complete_Unified_Colab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_2_source = [
    "# [CTRL-F: REPO-SETUP]\n",
    "# ==============================================================================\n",
    "# SECTION 0: ENVIRONMENT SETUP & PRE-TRAINED WEIGHTS SYNC\n",
    "# ==============================================================================\n",
    "import os, sys\n",
    "\n",
    "# Detect Colab environment and ensure clean directory paths\n",
    "if os.path.exists('/content'):\n",
    "    repo_cloned = False\n",
    "    for target_dir in ['/content/RoadDamageProject', '/content/AI-Road-Damage-Detection']:\n",
    "        if os.path.exists(os.path.join(target_dir, '.git')):\n",
    "            print(f'-> Found existing repository at {target_dir}. Pulling latest...')\n",
    "            %cd {target_dir}\n",
    "            !git pull origin main\n",
    "            repo_cloned = True\n",
    "            break\n",
    "    if not repo_cloned:\n",
    "        print('-> Fresh Colab session: Cloning repository into /content/RoadDamageProject...')\n",
    "        %cd /content\n",
    "        !git clone https://github.com/VY0TH/AI-Road-Damage-Detection.git RoadDamageProject\n",
    "        %cd /content/RoadDamageProject\n",
    "    \n",
    "    # Guarantee bidirectional symlink so both paths work seamlessly\n",
    "    if os.path.exists('/content/RoadDamageProject') and not os.path.exists('/content/AI-Road-Damage-Detection'):\n",
    "        !ln -s /content/RoadDamageProject /content/AI-Road-Damage-Detection\n",
    "    elif os.path.exists('/content/AI-Road-Damage-Detection') and not os.path.exists('/content/RoadDamageProject'):\n",
    "        !ln -s /content/AI-Road-Damage-Detection /content/RoadDamageProject\n",
    "    \n",
    "    %cd /content/RoadDamageProject\n",
    "    print(f'-> Active working directory confirmed: {os.getcwd()}')\n",
    "    !pip install -q ultralytics openpyxl imbalanced-learn torchvision\n",
    "\n",
    "import torch\n",
    "print(f'PyTorch Version: {torch.__version__} | GPU Available: {torch.cuda.is_available()}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'Active Device: {torch.cuda.get_device_name(0)}')\n",
    "else:\n",
    "    print('Active Device: CPU (All models support CPU fallback)')\n"
]

nb['cells'][2]['source'] = new_cell_2_source
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 2 successfully in", nb_path)
