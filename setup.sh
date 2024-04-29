echo "Hello anh em"
echo "Chung toi dang setup environment"
echo "==================="

pip install -q --upgrade pip
pip install -r -q requirements.txt
pip install -r -q /content/face-anti-spoofing/requirements.txt
pip install -r -q /kaggle/working/face-anti-spoofing/requirements.txt
echo "==================="
echo "Setup complete!"