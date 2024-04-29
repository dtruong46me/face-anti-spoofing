echo "Hello anh em"
echo "Chung toi dang setup environment"
echo "==================="

pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r /content/face-anti-spoofing/requirements.txt
pip install -q -r /kaggle/working/face-anti-spoofing/requirements.txt
echo "==================="
echo "Setup complete!"