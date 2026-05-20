# Note

* Please run train_save_point_v1.py to train, this file is based on train.py and support to save model and checkpoint after each epoch if there is better accuracy score, and load saved model and checkpoint to avoid training from zero-position after relaunch the program.

* Before you start training, you should read the original README.md to get base background knowledge, and then read [Records_of_Running_History-henry.md](Records_of_Running_History-henry.md) to get know how to prepare training data and run training with Python 3.11.

* The [Code_Learning-henry.md](Code_Learning-henry.md)/[Code_Learning-henry-en_us.md](Code_Learning-henry-en_us.md) explains detail code-level steps of Transformer alogrithem.

# Known Issues

* Since the size of training data is very small, only 30K, so after 50 epoches the accuracy score of validation will stop to increase, you should use more bigger training data instead.