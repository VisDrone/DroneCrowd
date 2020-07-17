import h5py
import torch
import shutil
import os
import glob

def load_train_data(root, train_step = 1):
    # training data
    train_path = os.path.join(root, 'train_data', 'images')
    train_list = []
    count = 0
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        if count % train_step == 0:
            train_list.append(img_path)
        count += 1
    train_list.sort()
    train_pair = []
    for idx in range(len(train_list)):
        img_pair = []
        pre_img = train_list[max(0, idx - 1)]
        cur_img = train_list[idx]
        seq_id1 = pre_img[-9:-7]
        seq_id2 = cur_img[-9:-7]
        if seq_id1 == seq_id2:
            img_pair.append(pre_img)
            img_pair.append(cur_img)
            train_pair.append(img_pair)
        else:
            img_pair.append(cur_img)
            img_pair.append(cur_img)
            train_pair.append(img_pair)

    return train_pair

def load_val_data(root, val_step = 10):
    # validation data
    val_path = os.path.join(root, 'val_data', 'images')
    val_list = []
    for img_path in glob.glob(os.path.join(val_path, '*.jpg')):
        val_list.append(img_path)
    val_list.sort()
    val_pair = []
    count = 0
    for idx in range(len(val_list)):
        if count % val_step == 0:
            img_pair = []
            pre_img = val_list[max(0, idx - 1)]
            cur_img = val_list[idx]
            seq_id1 = pre_img[-9:-7]
            seq_id2 = cur_img[-9:-7]
            if seq_id1 == seq_id2:
                img_pair.append(pre_img)
                img_pair.append(cur_img)
                val_pair.append(img_pair)
            else:
                img_pair.append(cur_img)
                img_pair.append(cur_img)
                val_pair.append(img_pair)
        count += 1

    return val_pair

def load_test_data(root, test_step = 5):
    test_path = os.path.join(root, 'test_data', 'images')
    test_list = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_list.append(img_path)
    test_list.sort()
    test_pair = []
    for idx in range(len(test_list)):
        img_pair = []
        pre_img = test_list[max(0, idx - test_step)]
        cur_img = test_list[idx]
        seq_id1 = pre_img[-9:-7]
        seq_id2 = cur_img[-9:-7]
        if seq_id1 == seq_id2:
            img_pair.append(pre_img)
            img_pair.append(cur_img)
            test_pair.append(img_pair)
        else:
            for id in range(test_step):
                pre_img = test_list[max(0, idx - (test_step - id - 1))]
                seq_id1 = pre_img[-9:-7]
                if seq_id1 == seq_id2:
                    img_pair.append(pre_img)
                    img_pair.append(cur_img)
                    test_pair.append(img_pair)
                    break
    return test_pair

def load_pretrained_model(model_name, model):
    print("=> loading checkpoint")
    checkpoint = torch.load(model_name)
    my_models = model.state_dict()
    pre_models = checkpoint['state_dict'].items()
    count = 0
    for layer_name, value in my_models.items():
        prelayer_name, pre_weights = pre_models[count]
        my_models[layer_name] = pre_weights
        count += 1
    model.load_state_dict(my_models)
    return model

def load_txt(fname):
    lines = fname.readlines()
    outs = []
    for i in range(len(lines)):
        predata = []
        for j in range(len(list(lines[0].split()))):
            predata.append(float(lines[i].split(' ')[j]))
        outs.append(predata)
    return outs


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')            
