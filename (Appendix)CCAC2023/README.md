### 环境配置
conda create -n CCAC python=3.8
conda activate CCAC
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pandas==1.5.3
pip install transformers==4.24.0
pip install opencv-python
pip install scikit-learn
pip install pytorch_lightning==1.8.3.post1

