export CUDA_VISIBLE_DEVICES=6
File_Name=cl
Model=gcn
Zero_Shot=unseen
Seed=0
Iteration_Num=50

python main.py  --filename ../data/datasets/${File_Name}.obo \
                --zero_shot_mode ${Zero_Shot} \
                --exp_path ../exp/${File_Name}/${Model}/${Zero_Shot}/seed_${Seed}/lr1_5-decay_0-iternum_${Iteration_Num}-graphrate_1 \
                --run_stage_1 \
                --test \
                --model ${Model} \
                --pretrain_model_path ../biobert \
                --initial_sparse_weight 0.5 \
                --lr  1e-5 \
                --weight_decay 0 \
                --iteration_num ${Iteration_Num} \
                --epoch_num 25 \
                --batch_size 16 \
                --with_graph --graph_rate 1.0 \
                --seed ${Seed} \
                --load_model \
                --output_file output_${File_Name}_${Model}_${Zero_Shot}.log \
                &

                #--output_file ../exp/${File_Name}/${Model}/${Zero_Shot}/seed_${Seed}/lr1_5-decay_0-iternum_${Iteration_Num}-graphrate_1/output.log \
                #--exp_path ../exp/${File_Name}/${Model}/${Zero_Shot}/seed_${Seed}/lr1_5-decay_0-iternum_${Iteration_Num}-graphrate_1 \
                
                #--exp_path /home/wzr/syn/exp/doid/gcn/unseen/lr1_5-decay_0-iternum_10-graphrate_1 \