import itertools

# warmup model
# params = ['IL', 'BATCH_SIZE', 'OPTIMIZER', 'LR']
# values = [[2,3], [16, 24, 32], ['adam', 'adamW'], [2e-5, 5e-6]]

params = ['BATCH_SIZE', 'LR', 'CONST']
values = [[16, 24, 32], [2e-5, 5e-6], ['0.1_0.1_0.1_0.1', '1_1_1_1', '3_3_3_3', '5_5_5_5', '10_10_10_10']]

# --checkpoint models/oie/grid_warmup/8/epoch=18_eval_acc=0.544.ckpt
cmd = 'python run.py --save SAVE --mode test --model_str bert-base-cased --task oie --epochs 21'+\
' --gpus 1 --batch_size BATCH_SIZE --optimizer adam --lr LR --iterative_layers 2'+\
' --constraints posm_hvc_hvr_hve --save_k 3 --accumulate_grad_batches 2'+\
' --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights CONST --val_check_interval 0.1'
grid_root = 'models/oie/grid_const/8'

model_num = 0
f = open('jobs/grid_const_test', 'w')
for value_list in itertools.product(*values):
    grid_cmd = cmd.replace('SAVE', grid_root+'/'+str(model_num))
    for param_i, param in enumerate(params):
        grid_cmd = grid_cmd.replace(param, str(value_list[param_i]))
    # if model_num % 6 == 0:
        # if model_num > 0:
        #     f.close()
        # f = open('jobs/grid_const_'+str(model_num), 'w')
    f.write(grid_cmd+'\n')
    model_num += 1
print(model_num)
f.close()