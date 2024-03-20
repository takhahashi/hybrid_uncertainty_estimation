cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=3e-05 training.per_device_train_batch_size\=16 +training.weight_decay\=0.1 +ue.margin\=10.0 +ue.lamb_intra\=0.02 ue.lamb\=0.001' task_configs=paradetox.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/paradetox/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=2 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0 +ue.margin\=0.025 +ue.lamb_intra\=0.01 ue.lamb\=1.0' task_configs=toxigen.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/toxigen/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=9 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.02 ue.lamb\=0.05' task_configs=implicithate.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/implicithate/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=3 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0 +ue.margin\=0.25 +ue.lamb_intra\=0.05 ue.lamb\=0.2' task_configs=jigsaw_toxic.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/jigsaw_toxic/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=6 training.learning_rate\=7e-06 training.per_device_train_batch_size\=32 +training.weight_decay\=0 +ue.margin\=0.25 +ue.lamb_intra\=0.2 ue.lamb\=0.05' task_configs=twitter_hso.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/twitter_hso/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 +ue.margin\=0.25 +ue.lamb_intra\=0.2 ue.lamb\=0.05' task_configs=sst5.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/sst5/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=3 training.learning_rate\=9e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.2 ue.lamb\=0.005' task_configs=amazon.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/amazon/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] script=run_glue_method_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=True ue.use_selective\=False model.model_name_or_path\=bert-base-uncased ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=9 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.02 ue.lamb\=0.05' task_configs=20newsgroups.yaml output_dir=../workdir/run_train_models_method_hp/bert_raw_sn/20newsgroups/0.2
Command: CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=../configs/paradetox.yaml  python run_glue_method_hp.py repeat=rep0 ue=mc do_ue_estimate=False +ue.use_spectralnorm=True ue.use_selective=False model.model_name_or_path=bert-base-uncased ue.calibrate=False ue.reg_type=raw data.validation_subsample=0.0 +data.eval_subsample=0.2 training=electra_base training.num_train_epochs=7 training.learning_rate=3e-05 training.per_device_train_batch_size=16 +training.weight_decay=0.1 +ue.margin=10.0 +ue.lamb_intra=0.02 ue.lamb=0.001 do_train=True do_eval=False do_ue_estimate=False seed=23419 output_dir=/content/hybrid_uncertainty_estimation/workdir/run_train_models_method_hp/bert_raw_sn/paradetox/0.2/models/paradetox/23419 hydra.run.dir=/content/hybrid_uncertainty_estimation/workdir/run_train_models_method_hp/bert_raw_sn/paradetox/0.2/models/paradetox/23419
