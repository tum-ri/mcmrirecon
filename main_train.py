from trainers import train_rl
from trainers import train
from configs import config_wnet
from configs import config_rl
import argparse





def main(args):
    '''
    Orchestrator method to load configs and call the corresponding train sequence

    '''
    if args.model == 'WNET':
        print('======WNET Training starts======')
        params = config_wnet.config

        params['resume'] = None  #'BestVal=False_R5WNet_dense=1_ikik_mse_lr=0.001_ep=50_complex=1_date=15-07-2020_21-04-30_checkpoint.pth.tar'
        params['num_edge_slices'] = 0
        params['edge_model'] = False
        params['num_workers'] = 4
        params['batch_size'] = 2
        params['architecture'] = 'iiii'
        params['domain'] = 'ki'
        params['lossFunction'] = 'msssim+vif'
        params['num_epochs'] = 100
        params['lr'] = 1e-3
        params['dense'] = True
        params['complex_net'] = True
        params['mask_flag'] = True
        params['verbose_delay'] = 4
        params['verbose_gap'] = 50
        params['complex_weight_init'] = False
        print('Params: ', params)
        train.train_net(params)
        
        # Complex Architecture --> Cloud Training
        params['num_workers'] = 8
        params['architecture'] = 'ii'
        params['domain'] = 'ki'
        params['lossFunction'] = 'vif'
        params['num_epochs'] = 35
        params['lr'] = 1e-4
        params['dense'] = True
        params['complex_net'] = True
        params['verbose_delay'] = 5
        params['verbose_gap'] = 50
        print('Params: ', params)
        train.train_net(params)
        
        '''
        -> Local Training (Maik)
        params['num_workers'] = 8
        params['architecture'] = 'ii'
        params['domain'] = 'ki'
        params['lossFunction'] = 'mse+vif'
        params['num_epochs'] = 35
        params['lr'] = 1e-4
        params['dense'] = True
        params['complex_net'] = True
        params['verbose_delay'] = 5
        params['verbose_gap'] = 50
        print('Params: ',params)
        train.train_net(params)
        
        
        -> Cloud Training (Fabi?)
        params['num_workers'] = 8
        params['architecture'] = 'iiii'
        params['domain'] = 'kk'
        params['lossFunction'] = 'vif'
        params['num_epochs'] = 20
        params['lr'] = 1e-4
        params['dense'] = True
        params['complex_net'] = False
        params['verbose_delay'] = 5
        params['verbose_gap'] = 50
        print('Params: ',params)
        train.train_net(params)
        '''
    
    if args.model == 'RL':
        print('======RL Training starts======')
        params = config_rl.config
        train_rl.train_net(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Pass argument to select model here \
                                        choose either of the following values {'WNET', 'RL'}")
    args = parser.parse_args()
    main(args)
