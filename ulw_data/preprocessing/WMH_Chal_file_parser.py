import os
from collections import defaultdict

def get_files_dict(ds_path, output_folder):
    files_dict = defaultdict(lambda: {})
    
        
    for fold in ['training', 'test']:
        for domain in ['Amsterdam_GE3T', 'Singapore', 'Utrecht', 'Amsterdam_GE1T5', 'Amsterdam_Philips_VU_PETMR_01']:
            domain_path = os.path.sep.join([ds_path, fold, domain])
            if os.path.exists(domain_path):
                num_ids = os.listdir(domain_path)
                for nid in num_ids:
                    ind_path = f'{domain_path}{os.path.sep}{nid}{os.path.sep}'
                    t1_path = f'{ind_path}pre{os.path.sep}T1.nii.gz'
                    flair_path = f'{ind_path}pre{os.path.sep}FLAIR.nii.gz'
                    wmh_path = f'{ind_path}wmh.nii.gz'
    
                    full_id = f'{fold}_{domain}_{nid}'
    
                    files_dict[full_id] = {
                        'FLAIR':flair_path,
                        'T1':t1_path,
                        'wmh':wmh_path,
                        'fold':fold,
                        'domain':domain,
                        'id':nid,
                        'out_path': f'{output_folder}{os.path.sep}{full_id}'
                    }

    return files_dict
