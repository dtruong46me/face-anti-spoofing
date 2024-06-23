echo "[+] INSTALL requirements.txt"

start=$(date +%s)
start_time=$(date +%s)

pip install -q --upgrade pip
pip install -q -r /kaggle/working/face-anti-spoofing/requirements.txt

end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
echo ""
echo "[+] Execution time: $execution_time seconds."


echo "............................................"
echo "[+] Visualize data"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/src/data/visualize_data.py\
--datapath "/kaggle/input/cv-dataset/final_data"\
--batch_size 32\
--num_images 5

echo "[+] Saved at: /kaggle/working/face-anti-spoofing/visualize_images.jpg"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
echo ""
echo "[+] Execution time: $execution_time seconds."


echo "............................................"
echo "[+] Train SE ResNeXT50"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_training.py\
--train_path "/kaggle/input/cv-dataset/final_data/train"\
--test_path "/kaggle/input/cv-dataset/final_data/valid"\
--batch_size 128\
--modelname "seresnext50"\
--wandb_token "c74fcec22fbb4be075a981b1f3db3f464b15b089"\
--wandb_runname "truong-resnext50"\
--num_classes 2\
--max_epochs 40
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Training time: ${hours}h ${minutes}m ${seconds}s"

echo "............................................"
echo "[+] Train MobileNetV3"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_training.py\
--train_path "/kaggle/input/cv-dataset/final_data/train"\
--test_path "/kaggle/input/cv-dataset/final_data/valid"\
--batch_size 128\
--modelname "mobilenetv3"\
--wandb_token "c74fcec22fbb4be075a981b1f3db3f464b15b089"\
--wandb_runname "truong-mobilenetv3"\
--num_classes 2\
--max_epochs 40
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Training time: ${hours}h ${minutes}m ${seconds}s"

echo "............................................"
echo "[+] Train FeatherNet"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_training.py\
--train_path "/kaggle/input/cv-dataset/final_data/train"\
--test_path "/kaggle/input/cv-dataset/final_data/valid"\
--batch_size 128\
--modelname "feathernet"\
--wandb_token "c74fcec22fbb4be075a981b1f3db3f464b15b089"\
--wandb_runname "truong-feathernet"\
--num_classes 2\
--max_epochs 40
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Training time: ${hours}h ${minutes}m ${seconds}s"


echo "............................................"
echo "[+] Eval SE ResNet50 on LCC_FASD_development"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_evaluation.py\
--test_path "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_development"\
--batch_size 128\
--model_checkpoint "/kaggle/working/checkpoint/seresnext50.ckpt"\
--modelname "seresnext50"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Evaluation time: ${hours}h ${minutes}m ${seconds}s"

echo "............................................"
echo "[+] Eval MobileNetV3 on LCC_FASD_development"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_evaluation.py\
--test_path "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_development"\
--batch_size 128\
--model_checkpoint "/kaggle/working/checkpoint/mobilenetv3.ckpt"\
--modelname "mobilenetv3"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Evaluation time: ${hours}h ${minutes}m ${seconds}s"

echo "............................................"
echo "[+] Eval FeatherNet on LCC_FASD_development"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_evaluation.py\
--test_path "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_development"\
--batch_size 128\
--model_checkpoint "/kaggle/working/checkpoint/feathernet.ckpt"\
--modelname "feathernet"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Evaluation time: ${hours}h ${minutes}m ${seconds}s"



echo "............................................"
echo "[+] Eval ResNet50 on LCC_FASD_evaluation"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_evaluation.py\
--test_path "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation"\
--batch_size 128\
--model_checkpoint "/kaggle/working/checkpoint/seresnext50.ckpt"\
--modelname "seresnext50"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Evaluation time: ${hours}h ${minutes}m ${seconds}s"

echo "............................................"
echo "[+] Eval MobileNetV3 on LCC_FASD_evaluation"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_evaluation.py\
--test_path "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation"\
--batch_size 128\
--model_checkpoint "/kaggle/working/checkpoint/mobilenetv3.ckpt"\
--modelname "mobilenetv3"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Evaluation time: ${hours}h ${minutes}m ${seconds}s"

echo "............................................"
echo "[+] Eval FeatherNet on LCC_FASD_evaluation"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/run_evaluation.py\
--test_path "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation"\
--batch_size 128\
--model_checkpoint "/kaggle/working/checkpoint/feathernet.ckpt"\
--modelname "feathernet"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo ""
echo "[+] Evaluation time: ${hours}h ${minutes}m ${seconds}s"



echo "............................................"
echo "[+] RestNext50 Predict sample"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/predict_sample.py\
--model_checkpoint "/kaggle/working/checkpoint/seresnext50.ckpt"\
--image "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_development/real/UNKNOWN_id152_s0_105.png"\
--modelname "seresnext50"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
echo ""
echo "[+] Inference time: $execution_time seconds."

echo "............................................"
echo "[+] Predict sample"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/predict_sample.py\
--model_checkpoint "/kaggle/working/checkpoint/mobilenetv3.ckpt"\
--image "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_development/real/UNKNOWN_id152_s0_105.png"\
--modelname "mobilenetv3"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
echo ""
echo "[+] Inference time: $execution_time seconds."

echo "............................................"
echo "[+] Featernet Predict sample"
start_time=$(date +%s)
python /kaggle/working/face-anti-spoofing/predict_sample.py\
--model_checkpoint "/kaggle/working/checkpoint/feathernet.ckpt"\
--image "/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_development/real/UNKNOWN_id152_s0_105.png"\
--modelname "feathernet"
end_time=$(date +%s)
execution_time = $(($end_time - $start_time))
echo ""
echo "[+] Inference time: $execution_time seconds."


echo "............................................"
end=$(date +%s)
execution_time = $(($end - $start))
hours=$((execution_time // 3600))
minutes=$((execution_time % 3600 // 60))
seconds=$((execution_time % 60))
echo "[+] Total time: ${hours}h ${minutes}m ${seconds}s"