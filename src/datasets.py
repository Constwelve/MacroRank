from functools import reduce
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import pylab
import os
from torch_scatter import scatter
from PIL import Image
from torch_geometric.data import Dataset, download_url, Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torchvision import transforms
import pdb
from tqdm import tqdm

def draw_rect(coord, size, path = 'figs'):
    from matplotlib.collections import PatchCollection
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, aspect='equal')
    plt.axis('off')
    plt.xlim(xmax=1.2,xmin=-0.2)
    plt.ylim(ymax=1.2,ymin=-0.2)

    c , s  = coord, size
    #patches = [matplotlib.patches.Rectangle((x, y),w, h, alpha=0.2,color='blue') for x,y,w,h in zip(coord[0], coord[1], size[0], size[1])]
    #ax.add_collection(PatchCollection(patches))
    [ax.add_patch(plt.Rectangle((x, y),w, h, alpha=0.2,facecolor='blue')) for x,y,w,h in zip(c[0], c[1], s[0], s[1])]
    fig.savefig(os.path.join(path,"draw.png"),bbox_inches='tight')
    plt.close(fig)
    plt.cla()
    plt.clf()

def GasussianKernel(size=41):
    if size%2 == 0:
        size = size + 1 

    cx = size//2 
    cy = size//2
    matrix = torch.zeros((size,size),dtype=torch.float)
    for i in range(0,size):
        for j in range(0,size):
            matrix[i,j] = np.exp(-np.abs(i-cx)/10-np.abs(j-cy)/10)
    return matrix

class GaussianBlur(nn.Module):
    def __init__(self,size=41):
        super(GaussianBlur,self).__init__()
        self.size = size
        kernel = GasussianKernel(size).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel,requires_grad=False)
    def forward(self,x):
        x = F.conv2d(x.unsqueeze(0).unsqueeze(0),self.weight,padding=self.size//2)
        return x


class ISPD15Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ISPD15Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['des_perf_a', 'edit_dist_a', 'fft_a','fft_b','matrix_mult_a','matrix_mult_b','matrix_mult_c','pci_bridge32_a','pci_bridge32_b']

    @property
    def processed_file_names(self):
        return ['data_%d.pt'%i for i in range(0,9)]

    def process(self):
        i = 0
        for design in self.raw_file_names:
            path = osp.join(self.raw_dir,design)
            net_path = osp.join(path,'nets.txt')
            nodes_path = osp.join(path,'node_feature.txt')
            name_path = osp.join(path,'names.txt')
            pos_path = osp.join(path,'pos.txt')
            graph_lab_path = osp.join(path,'graph_labels.txt')
            pin_path = osp.join(path,'pins.txt')
            region_path = osp.join(path,'region.txt')
            # Read data from `raw_path`.
            nets = np.array(pd.read_table(net_path,header=None))
            node_lab = np.array(pd.read_table(nodes_path,header=None))
            pin_feature = np.array(pd.read_table(pin_path,header=None))[:,2:]
            xl,yl,xh,yh = np.loadtxt(region_path)
            # normalize
            node_lab[:,0] /= xh-xl
            node_lab[:,1] /= yh-yl
            pin_feature[:,0] /= xh-xl
            pin_feature[:,1] /= yh-yl

            x = torch.tensor(node_lab, dtype=torch.float)
            edge_index = torch.tensor(nets.T, dtype=torch.long)
            pin_feature = torch.tensor(pin_feature,dtype=torch.float)
            data = Data(x=x, edge_index=edge_index,y=random.random(),pin_feature=pin_feature)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data





class CNNSet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, classes=20):
        self.root = root
        self.tot_file_num = None # int
        self.file_num = None # dict, file nums for each design
        self.ptr = None
        self.classes = classes
        self.num_bins = 224
        self.bin_size = 1./224
        self.netlist = {}
        self.labels = ['wl','vias','short']
        self.label_key = None

        names_path = osp.join(self.root,'raw','all.names')
        names = np.loadtxt(names_path,dtype=str)
        self.rawfn = names.tolist()

        names_path = osp.join(self.root,'raw','train.names')
        names = np.loadtxt(names_path,dtype=str)
        self.trainfn = names.tolist()

        names_path = osp.join(self.root,'raw','test.names')
        names = np.loadtxt(names_path,dtype=str)
        self.testfn = names.tolist()
        self.weight  = {}
        super(CNNSet, self).__init__(root, transform, pre_transform)
        self.balence_weight()

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_cnn')

    @property
    def raw_file_names(self):
        return self.rawfn
    
    @property
    def train_file_names(self):
        return self.trainfn

    @property
    def test_file_names(self):
        return self.testfn

    @property
    def num_features(self):
        return 3

    @property
    def num_classes(self):
        return int(self.classes)

    @property
    def processed_file_names(self):
        if self.tot_file_num is None:
            self.tot_file_num = 0
            self.file_num = {}
            self.ptr = {}
            for design in self.raw_file_names:
                path = osp.join(self.raw_dir,design)
                name_path = osp.join(path,'names.txt')
                names = np.array(pd.read_table(name_path,header=None)).reshape(-1)
                self.tot_file_num += names.shape[0]
                self.file_num[design] = names.shape[0]
            self.ptr[self.raw_file_names[0]] = 0
            for i in range(1,int(len(self.raw_file_names))):
                self.ptr[self.raw_file_names[i]] = self.ptr[self.raw_file_names[i-1]] + self.file_num[self.raw_file_names[i-1]]
        return ['data_%d.pt'%i for i in range(0,self.tot_file_num)]

    def balence_weight(self):
        wlist = []
        for label in self.labels:
            wt = {}
            weight = {}
            
            for design in self.raw_file_names:
                wt[design] = []
            id2design = lambda idx : [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]
            for i, data in enumerate(self):
                y = getattr(data,label)
                design = id2design(i)
                if design in self.raw_file_names:
                    wt[design].append(y)
            for design in self.raw_file_names:
                weight[design] = 1. / np.var(wt[design])
                wlist.append(weight[design])
            self.weight[label] = weight
        scaler = 1. / np.mean(wlist)
        for design in self.raw_file_names:
            for label in self.labels:
                self.weight[label][design] *= scaler
        return weight


    def process(self):

        i = 0
        for design in self.raw_file_names:
        #for design in ['mgc_matrix_mult_a']:
            # paths
            path = osp.join(self.raw_dir,design)
            size_path = osp.join(path,'node_size.txt')
            name_path = osp.join(path,'names.txt')
            pos_root = osp.join(path,'node_pos')
            #wl_path = osp.join(path,'wl.txt')
            pin_path = osp.join(path,'pins.txt')
            region_path = osp.join(path,'region.txt')
            golden_path = osp.join(path,'golden.txt')
            dist_path = osp.join(path,'dist2macro.txt')
            macro_path = osp.join(path,'macro_index.txt')
            hpwl_path = osp.join(path,'hpwl.txt')
            meta_path = osp.join(path,'meta.txt')
            label_path = osp.join(path,'labels.txt')
            fixed_path = osp.join(path,'fixed_node_index.txt')
            # Read data from `raw_path`.
            golden = np.loadtxt(golden_path)
            pins = np.loadtxt(pin_path)
            size = np.loadtxt(size_path)
            incidence = pins[:,:2]
            pin_feature = pins[:,2:]
            xl,yl,xh,yh = np.loadtxt(region_path)

            macro_index = torch.tensor(np.loadtxt(macro_path),dtype=torch.long)
            if osp.exists(dist_path):
                dist2macro = np.loadtxt(dist_path)
            else:
                dist2macro = np.ones((size.shape[0], macro_index.shape[0]))
            names = np.loadtxt(name_path,dtype=int)
            #with open(wl_path,'r') as f:
            #    rWLs = np.array([float(line) if line != '\n' else 0 for line in f.readlines()])
            
            hpwls = np.loadtxt(hpwl_path)
            meta_data = np.loadtxt(meta_path)
            labels = np.loadtxt(label_path)
            fixed_index = torch.from_numpy(np.loadtxt(fixed_path)).long()
            rWLs = labels[:,0]
            vias = labels[:,1]
            short = labels[:,2]
            score = labels[:,3]

            meta_data[5] = meta_data[5]/(yh-yl)
            meta_data[8] = meta_data[8]/(yh-yl)/(xh-xl)
            meta_data[9] = meta_data[9]/(yh-yl)/(xh-xl)
            meta_data[10] = meta_data[10]/(yh-yl)/(xh-xl)

            meta_data = torch.from_numpy(meta_data).float()
            fixed_cell_index = [i for i in fixed_index if i not in macro_index]
            fixed_macro_index = [i for i in fixed_index if i in macro_index]
            # stastics
            # rel = (rWLs[names]).tolist()
            # pylab.hist(rel,20)
            # pylab.xlabel('Range')
            # pylab.ylabel('Count')
            # pylab.savefig('stat/{}.png'.format(design+'_golden'))
            # pylab.cla()
            # normalize
            size[:,0] = size[:,0]/(xh-xl)
            size[:,1] = size[:,1]/(yh-yl)
            pin_feature[:,0] = pin_feature[:,0]/(xh-xl)
            pin_feature[:,1] = pin_feature[:,1]/(yh-yl)
            # std
            rWLs = rWLs/(xh-xl+yh-yl)*2
            rWLs = rWLs/1000.0

            hpwls = hpwls/(xh-xl+yh-yl)*2
            hpwls = hpwls/1000.0

            orWLs = rWLs.copy()

            cell_size = torch.tensor(size, dtype=torch.float)
            edge_index = torch.tensor(incidence.T, dtype=torch.long)
            pins = torch.tensor(pin_feature,dtype=torch.float)
            gold = torch.tensor(golden,dtype=torch.float)
            weight = torch.tensor(dist2macro,dtype=torch.float)
            weight = 1/weight
            summ = weight.sum(dim=-1).view(-1,1).repeat(1,int(len(macro_index)))
            weight = weight/summ
            B = scatter(cell_size.new_ones(edge_index.size(1)),
                        edge_index[1], dim=0, dim_size=edge_index[1].max()+1, reduce='sum')

            B  = torch.index_select(B,dim=-1,index=edge_index[1]).clamp(0,50)
            
            
            #weight = torch.softmax(1/weight,dim=-1)
            for name in tqdm(names):
                
                #if osp.exists(osp.join(self.processed_dir, 'data_{}.pt'.format(i))):
                #    i += 1
                #    continue
                if hpwls[name] == 0:
                    print('{}-{}'.format(design,name))
                pos_path = osp.join(pos_root,'%d.txt'%name)
                node_pos = torch.tensor(np.loadtxt(pos_path),dtype=torch.float) 

                node_pos[:,0] = (node_pos[:,0]-xl)/(xh-xl)
                node_pos[:,1] = (node_pos[:,1]-yl)/(yh-yl)

                macro_pos = node_pos[macro_index]
                # draw_rect(macro_pos.T, size[macro_index].T, '.')
                # exit(0)
                density = []

                ox = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                oy = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                for idx in macro_index:
                    pos = node_pos[idx]
                    size = cell_size[idx]
                    ox = torch.arange(0,1,self.bin_size,dtype=float).view(1,-1).repeat(self.num_bins,1)
                    oy = torch.arange(0,1,self.bin_size,dtype=float).view(-1,1).repeat(1,self.num_bins)

                    ox = torch.clamp((size[0]/2 + self.bin_size/2 - torch.abs(pos[0] - ox + size[0]/2 - self.bin_size/2)) / self.bin_size,0,1)
                    oy = torch.clamp((size[1]/2 + self.bin_size/2 - torch.abs(pos[1] - oy + size[1]/2 - self.bin_size/2)) / self.bin_size,0,1)

                    density.append((ox * oy).view(self.num_bins,self.num_bins,1))

                density = torch.cat(density,dim = -1)

                density_map = density.sum(dim=-1)

                pin_density = torch.zeros_like(density_map).view(-1)
                cnt_density = torch.zeros_like(density_map).view(-1)


                all_pin_pos = ((torch.index_select(node_pos,dim=0,index=edge_index[0]) + pins) /self.bin_size).long().clamp(0, self.num_bins - 1)
                def dd2d(index):
                    return index[:,1] * self.num_bins + index[:,0]
                pin_mask = torch.zeros(all_pin_pos.shape[0]).bool()
                for w in macro_index:
                    pin_mask |= (edge_index[0]==w)
                for w in fixed_cell_index:
                    pin_mask |= (edge_index[0]==w)    
                pin_pos = all_pin_pos[pin_mask] 
                indx = dd2d(pin_pos)
                pin_density = pin_density + scatter(all_pin_pos.new_ones(pin_pos.shape[0]), \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')
                    
                cnt_density = cnt_density + scatter(B[pin_mask], \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')

                pin_density /= pin_density.max()
                cnt_density /= cnt_density.max()

                pin_density = pin_density.view(self.num_bins,self.num_bins)
                cnt_density = cnt_density.view(self.num_bins,self.num_bins)

                pic = torch.cat([density_map.view(1,1,self.num_bins,self.num_bins),pin_density.view(1,1,self.num_bins,self.num_bins),cnt_density.view(1,1,self.num_bins,self.num_bins)],dim=1)
                net_pic = torch.clamp(cnt_density,0,1)
                #im = Image.fromarray(np.uint8((net_pic*255).view(self.num_bins,self.num_bins).numpy()))
                #im.convert('L')
                #im.save('test.jpg') 
                #exit(0)
                y = torch.tensor(1.0,dtype=torch.long)
                if y.ndim <= 1:
                    y = y.view(1,-1)
                data = Data(x=torch.zeros(1), 
                            edge_index=torch.zeros(2).view(2,1).long(),
                            y=torch.zeros(1),
                            wl=torch.tensor(orWLs[name],dtype=torch.float),
                            density = pic.float(),
                            vias = torch.tensor(vias[name],dtype=torch.float),
                            short = torch.tensor(short[name],dtype=torch.float),
                            score = torch.tensor(score[name],dtype=torch.float),
                            meta_feature=meta_data,
                            hpwl=torch.tensor(hpwls[name],dtype=torch.float))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        #data.density = data.density.view(1,1,self.num_bins,self.num_bins)
        data.short = data.short/100000
        data.score = data.score/100000000
        data.vias = data.vias/100000
        return data




class LITESet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, classes=20):
        self.tot_file_num = None # int
        self.file_num = None # dict, file nums for each design
        self.ptr = None
        self.classes = classes
        self.num_bins = 224
        self.bin_size = 1./224
        self.label_key = None
        self.netlist = {}
        self.labels = ['wl','vias','short']
        self.weight = {}
        super(LITESet, self).__init__(root, transform, pre_transform)
        for design in self.raw_file_names:
            self.netlist[design] = torch.load(osp.join(self.processed_dir, '{}.pt'.format(design)))
        self.balence_weight()

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_lite')

    @property
    def raw_file_names(self):
        names_path = osp.join(self.root,'raw','all.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return names.tolist()
    
    @property
    def train_file_names(self):
        names_path = osp.join(self.root,'raw','train.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return [name for name in names]

    @property
    def test_file_names(self):
        names_path = osp.join(self.root,'raw','test.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return [name for name in names]

    @property
    def num_features(self):
        return self[0].x.size(1) + self[0].cell_pos.size(1) + self[0].pin_feature.size(1) + 1 # macro label

    @property
    def num_classes(self):
        return int(self.classes)

    @property
    def processed_file_names(self):
        if self.tot_file_num is None:
            self.tot_file_num = 0
            self.file_num = {}
            self.ptr = {}
            for design in self.raw_file_names:
                path = osp.join(self.raw_dir,design)
                name_path = osp.join(path,'names.txt')
                names = np.array(pd.read_table(name_path,header=None)).reshape(-1)
                self.tot_file_num += names.shape[0]
                self.file_num[design] = names.shape[0]
            self.ptr[self.raw_file_names[0]] = 0
            for i in range(1,int(len(self.raw_file_names))):
                self.ptr[self.raw_file_names[i]] = self.ptr[self.raw_file_names[i-1]] + self.file_num[self.raw_file_names[i-1]]
        return ['data_%d.pt'%i for i in range(0,self.tot_file_num)]


    def process(self):

        i = 0
        for design in self.raw_file_names:
            # paths
            path = osp.join(self.raw_dir,design)
            size_path = osp.join(path,'node_size.txt')
            name_path = osp.join(path,'names.txt')
            pos_root = osp.join(path,'node_pos')
            #wl_path = osp.join(path,'wl.txt')
            pin_path = osp.join(path,'pins.txt')
            region_path = osp.join(path,'region.txt')
            golden_path = osp.join(path,'golden.txt')
            dist_path = osp.join(path,'dist2macro.txt')
            macro_path = osp.join(path,'macro_index.txt')
            hpwl_path = osp.join(path,'hpwl.txt')
            meta_path = osp.join(path,'meta.txt')
            label_path = osp.join(path,'labels.txt')
            fixed_path = osp.join(path,'fixed_node_index.txt')
            # Read data from `raw_path`.
            golden = np.loadtxt(golden_path)
            pins = np.loadtxt(pin_path)
            size = np.loadtxt(size_path)
            incidence = pins[:,:2]
            pin_feature = pins[:,2:]
            xl,yl,xh,yh = np.loadtxt(region_path)
            
            
            macro_index = torch.tensor(np.loadtxt(macro_path),dtype=torch.long)
            if osp.exists(dist_path):
                dist2macro = np.loadtxt(dist_path)
            else:
                dist2macro = np.ones((size.shape[0], macro_index.shape[0]))
            names = np.loadtxt(name_path,dtype=int)
            #with open(wl_path,'r') as f:
            #    rWLs = np.array([float(line) if line != '\n' else 0 for line in f.readlines()])
            
            hpwls = np.loadtxt(hpwl_path)
            meta_data = np.loadtxt(meta_path)
            labels = np.loadtxt(label_path)
            fixed_index = torch.from_numpy(np.loadtxt(fixed_path)).long()
            fixed_cell_index = torch.tensor([i for i in fixed_index if i not in macro_index]).long()
            
            rWLs = labels[:,0]
            vias = labels[:,1]
            short = labels[:,2]
            score = labels[:,3]

            meta_data[5] = meta_data[5]/(yh-yl)
            meta_data[8] = meta_data[8]/(yh-yl)/(xh-xl)
            meta_data[9] = meta_data[9]/(yh-yl)/(xh-xl)
            meta_data[10] = meta_data[10]/(yh-yl)/(xh-xl)

            meta_data = torch.from_numpy(meta_data).float()
            # stastics
            # rel = (rWLs[names]).tolist()
            # pylab.hist(rel,20)
            # pylab.xlabel('Range')
            # pylab.ylabel('Count')
            # pylab.savefig('stat/{}.png'.format(design+'_golden'))
            # pylab.cla()
            # normalize
            size[:,0] = size[:,0]/(xh-xl)
            size[:,1] = size[:,1]/(yh-yl)
            pin_feature[:,0] = pin_feature[:,0]/(xh-xl)
            pin_feature[:,1] = pin_feature[:,1]/(yh-yl)
            # std
            rWLs = rWLs/(xh-xl+yh-yl)*2
            rWLs = rWLs/1000.0

            hpwls = hpwls/(xh-xl+yh-yl)*2
            hpwls = hpwls/1000.0

            orWLs = rWLs.copy()

            cell_size = torch.tensor(size, dtype=torch.float)
            edge_index = torch.tensor(incidence.T, dtype=torch.long)
            pins = torch.tensor(pin_feature,dtype=torch.float)
            gold = torch.tensor(golden,dtype=torch.float)
            weight = torch.tensor(dist2macro,dtype=torch.float)
            weight = 1/weight
            summ = weight.sum(dim=-1).view(-1,1).repeat(1,int(len(macro_index)))
            weight = weight/summ
            B = scatter(cell_size.new_ones(edge_index.size(1)),
                        edge_index[1], dim=0, dim_size=edge_index[1].max()+1, reduce='sum')

            B  = torch.index_select(B,dim=-1,index=edge_index[1]).clamp(0,50)
            
            #weight = torch.softmax(1/weight,dim=-1)
            # netlist is the same
            data = Data(x=cell_size, 
                    edge_index=edge_index,
                    pin_feature=pins,
                    macro_index=macro_index,
                    fixed_cell_index = fixed_cell_index)
            torch.save(data, osp.join(self.processed_dir, '{}.pt'.format(design)))
            for name in tqdm(names):
                if osp.exists(osp.join(self.processed_dir, 'data_{}.pt'.format(i))):
                    i += 1
                    continue
                if hpwls[name] == 0:
                    print('{}-{}'.format(design,name))
                pos_path = osp.join(pos_root,'%d.txt'%name)
                node_pos = torch.tensor(np.loadtxt(pos_path),dtype=torch.float) 

                node_pos[:,0] = (node_pos[:,0]-xl)/(xh-xl)
                node_pos[:,1] = (node_pos[:,1]-yl)/(yh-yl)

                macro_pos = node_pos[macro_index]

                fake_pos = torch.matmul(weight,macro_pos)
                fake_pos[fixed_index] = node_pos[fixed_index]
        
                density = []

                ox = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                oy = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                for idx in macro_index:
                    pos = node_pos[idx]
                    size = cell_size[idx]
                    ox = torch.arange(0,1,self.bin_size,dtype=float).view(1,-1).repeat(self.num_bins,1)
                    oy = torch.arange(0,1,self.bin_size,dtype=float).view(-1,1).repeat(1,self.num_bins)

                    ox = torch.clamp((size[0]/2 + self.bin_size/2 - torch.abs(pos[0] - ox + size[0]/2 - self.bin_size/2)) / self.bin_size,0,1)
                    oy = torch.clamp((size[1]/2 + self.bin_size/2 - torch.abs(pos[1] - oy + size[1]/2 - self.bin_size/2)) / self.bin_size,0,1)

                    density.append((ox * oy).view(self.num_bins,self.num_bins,1))

                density = torch.cat(density,dim = -1)
                density_map = density.sum(dim=-1)

                pin_density = torch.zeros_like(density_map).view(-1)
                cnt_density = torch.zeros_like(density_map).view(-1)


                all_pin_pos = ((torch.index_select(node_pos,dim=0,index=edge_index[0]) + pins) /self.bin_size).long().clamp(0, self.num_bins - 1)
                def dd2d(index):
                    return index[:,1] * self.num_bins + index[:,0]
                pin_mask = torch.zeros(all_pin_pos.shape[0]).bool()
                for pidx in macro_index:
                    pin_mask |= (edge_index[0]==pidx)
                for pidx in fixed_cell_index:
                    pin_mask |= (edge_index[0]==pidx)    
                pin_pos = all_pin_pos[pin_mask] 
                indx = dd2d(pin_pos)

                pin_density = pin_density + scatter(all_pin_pos.new_ones(pin_pos.shape[0]), \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')
                    
                cnt_density = cnt_density + scatter(B[pin_mask], \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')

                pin_density /= pin_density.max()
                cnt_density /= cnt_density.max()

                pin_density = pin_density.view(self.num_bins,self.num_bins)
                cnt_density = cnt_density.view(self.num_bins,self.num_bins)

                pic = torch.cat([density_map.view(1,1,self.num_bins,self.num_bins),pin_density.view(1,1,self.num_bins,self.num_bins),cnt_density.view(1,1,self.num_bins,self.num_bins)],dim=1)

                y = torch.tensor(1.0,dtype=torch.long)
                if y.ndim <= 1:
                    y = y.view(1,-1)
                data = Data(x=torch.tensor(1,dtype=torch.float), 
                            edge_index=torch.tensor([1,1],dtype=torch.long),
                            y=y,
                            pin_feature=torch.tensor(1,dtype=torch.float),
                            macro_index=torch.tensor(1,dtype=torch.float),
                            fixed_cell_index = torch.tensor(1,dtype=torch.float),
                            golden=gold,
                            wl=torch.tensor(orWLs[name],dtype=torch.float),
                            density = density.float(),
                            pic = pic.float(),
                            vias = torch.tensor(vias[name],dtype=torch.float),
                            short = torch.tensor(short[name],dtype=torch.float),
                            score = torch.tensor(score[name],dtype=torch.float),
                            meta_feature=meta_data,
                            cell_pos=macro_pos,
                            fake_pos=macro_pos,
                            hpwl=torch.tensor(hpwls[name],dtype=torch.float))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def balence_weight(self):
        wlist = []
        for label in self.labels:
            wt = {}
            weight = {}
            
            for design in self.raw_file_names:
                wt[design] = []
            id2design = lambda idx : [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]
            for i, data in enumerate(self):
                y = getattr(data,label)
                design = id2design(i)
                if design in self.raw_file_names:
                    wt[design].append(y)
            for design in self.raw_file_names:
                weight[design] = 1. / np.var(wt[design])
                wlist.append(weight[design])
            self.weight[label] = weight
        scaler = 1. / np.mean(wlist)
        for design in self.raw_file_names:
            for label in self.labels:
                self.weight[label][design] *= scaler
        return weight



    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        data.short = data.short/100000
        data.score = data.score/100000000
        data.vias = data.vias/100000
        data.design = [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]

        data.x=self.netlist[data.design].x
        data.edge_index=self.netlist[data.design].edge_index
        data.pin_feature=self.netlist[data.design].pin_feature
        data.macro_index=self.netlist[data.design].macro_index
        data.fixed_cell_index = self.netlist[data.design].fixed_cell_index
        data.all = torch.cat([data.wl.view(1,1),data.vias.view(1,1),data.short.view(1,1)],dim=1)
        return data

def cluster():
    data = Set[0].cuda()
    target_density = 1.
    edge_index = data.edge_index
    macro_index = data.macro_index
    fixed_pin = data.fixed_cell_index
    x = data.x
    num_nodes = x.shape[0]
    num_edges = edge_index[1].max() + 1
    B = scatter(x.new_ones(edge_index.size(1)).long(),
                        edge_index[1], dim=0, dim_size=num_edges)
    X = torch.index_select(x,dim=0,index=edge_index[0])
    X = X[:,0] * X[:,1]
    mean_area = X.mean() * 10
    X[macro_index.max():] += mean_area # donot merge fixed pin
    S = scatter(X,edge_index[1], dim=0, dim_size=num_edges)
    
    K =  B * 10 + S
    v, idxs = torch.sort(K)

    def merge_an_edge(idx):
        merge_mask = (edge_index[1]==idx)
        pins2merge = edge_index[:,merge_mask]
        nodes2merge = pins2merge[0]
        merge_size = S[nodes2merge].sum() / target_density
        new_node_idx = 0


    
    
    
    def merge(deg, edge_index, size):
        pdb.set_trace()
        merge_mask  = (B == deg) & (S < mean_area)
        edge2merge = torch.arange(0,num_edges)[merge_mask].to(edge_index.device)
        rm_edges =  merge_mask.sum()
        old_egde_num = edge_index[1].max() + 1
        old_node_num = size.shape[0]
        new_egde_num = old_egde_num - rm_edges
        new_node_num = old_node_num - (deg - 1) * rm_edges
        old_edge2new_edge_map = edge_index.new_ones(old_egde_num) * (-1)
        old_edge2new_edge_map[~merge_mask] = torch.arange(0,new_egde_num).to(edge_index.device)

        rmd_edge2new_node = edge_index.new_ones(old_egde_num) * (-1)
        rmd_edge2new_node[merge_mask] = rmd_edge2new_node[merge_mask] + 2
        rmd_edge2new_node[merge_mask] = torch.arange(0, rm_edges).to(edge_index.device)

        rmd_node2new_node = torch.index_select(rmd_edge2new_node,dim=0,index=edge_index[1])
        
        old_node2new_node_map = edge_index.new_ones(old_node_num) * (-1)
        np.savetxt('1.txt',rmd_node2new_node.cpu().numpy())
        old_node2new_node_map = old_node2new_node_map + scatter(rmd_node2new_node,index=edge_index[0],dim_size=old_node_num, reduce='max')
        np.savetxt('1.txt',old_node2new_node_map.cpu().numpy())
        old_node2new_node_map[old_node2new_node_map<0] = torch.arange(rm_edges, new_node_num).to(edge_index.device)
        # map rmd node to the lower sapce
        new_size = torch.gather(size,index=old_node2new_node_map)

        new_mask = (old_edge2new_edge_map >= 0)
        new_edge_inde = torch.cat(old_node2new_node_map[edge_index[0][new_mask]], old_edge2new_edge_map[edge_index[1][new_mask]])
        new_edge_inde = torch.unique(new_edge_inde,dim=0)

        return edge_index


    merge(2,edge_index,S)

    a = B[idxs]
    b = S[idxs]
    R = torch.cat([a.view(-1,1),b.view(-1,1)],dim=-1)
    np.savetxt("results.txt",R.cpu().numpy())


class ClusterSet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, classes=20):
        self.tot_file_num = None # int
        self.file_num = None # dict, file nums for each design
        self.ptr = None
        self.classes = classes
        self.num_bins = 224
        self.bin_size = 1./224
        self.label_key = None
        self.netlist = {}
        self.labels = ['wl','vias','short']
        self.weight = {}
        super(ClusterSet, self).__init__(root, transform, pre_transform)
        for design in self.raw_file_names:
            self.netlist[design] = torch.load(osp.join(self.processed_dir, '{}.pt'.format(design)))
        #self.balence_weight()

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_cluster')

    @property
    def raw_file_names(self):
        names_path = osp.join(self.root,'raw','all.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return names.tolist()
    
    @property
    def train_file_names(self):
        names_path = osp.join(self.root,'raw','train.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return [name for name in names]

    @property
    def test_file_names(self):
        names_path = osp.join(self.root,'raw','test.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return [name for name in names]

    @property
    def num_features(self):
        return self[0].x.size(1) + self[0].cell_pos.size(1) + self[0].pin_feature.size(1) + 1 # macro label

    @property
    def num_classes(self):
        return int(self.classes)

    @property
    def processed_file_names(self):
        if self.tot_file_num is None:
            self.tot_file_num = 0
            self.file_num = {}
            self.ptr = {}
            for design in self.raw_file_names:
                path = osp.join(self.raw_dir,design)
                name_path = osp.join(path,'names.txt')
                names = np.array(pd.read_table(name_path,header=None)).reshape(-1)
                self.tot_file_num += names.shape[0]
                self.file_num[design] = names.shape[0]
            self.ptr[self.raw_file_names[0]] = 0
            for i in range(1,int(len(self.raw_file_names))):
                self.ptr[self.raw_file_names[i]] = self.ptr[self.raw_file_names[i-1]] + self.file_num[self.raw_file_names[i-1]]
        return ['data_%d.pt'%i for i in range(0,self.tot_file_num)]


    def process(self):

        i = 0
        for design in self.raw_file_names:
            # paths
            path = osp.join(self.raw_dir,design)
            size_path = osp.join(path,'node_size.txt')
            name_path = osp.join(path,'names.txt')
            pos_root = osp.join(path,'node_pos')
            #wl_path = osp.join(path,'wl.txt')
            pin_path = osp.join(path,'pins.txt')
            region_path = osp.join(path,'region.txt')
            golden_path = osp.join(path,'golden.txt')
            #dist_path = osp.join(path,'dist2macro.txt')
            macro_path = osp.join(path,'macro_index.txt')
            hpwl_path = osp.join(path,'hpwl.txt')
            meta_path = osp.join(path,'meta.txt')
            label_path = osp.join(path,'labels.txt')
            #fixed_path = osp.join(path,'fixed_node_index.txt')
            # Read data from `raw_path`.
            golden = np.loadtxt(golden_path)
            pins = np.loadtxt(pin_path)
            size = np.loadtxt(size_path)
            incidence = pins[:,:2]
            pin_feature = pins[:,2:]
            xl,yl,xh,yh = np.loadtxt(region_path)
            
            
            macro_index = torch.tensor(np.loadtxt(macro_path),dtype=torch.long)
            #if osp.exists(dist_path):
            #    dist2macro = np.loadtxt(dist_path)
            #else:
            #    dist2macro = np.ones((size.shape[0], macro_index.shape[0]))
            names = np.loadtxt(name_path,dtype=int)
            #with open(wl_path,'r') as f:
            #    rWLs = np.array([float(line) if line != '\n' else 0 for line in f.readlines()])
            
            hpwls = np.loadtxt(hpwl_path)
            meta_data = np.loadtxt(meta_path)
            labels = np.loadtxt(label_path)
            fixed_index = macro_index#torch.from_numpy(np.loadtxt(fixed_path)).long()
            fixed_cell_index = torch.tensor([i for i in fixed_index if i not in macro_index]).long()
            
            rWLs = labels[:,0]
            vias = labels[:,1]
            short = labels[:,2]
            score = labels[:,3]

            meta_data[5] = meta_data[5]/(yh-yl)
            meta_data[8] = meta_data[8]/(yh-yl)/(xh-xl)
            meta_data[9] = meta_data[9]/(yh-yl)/(xh-xl)
            meta_data[10] = meta_data[10]/(yh-yl)/(xh-xl)

            meta_data = torch.from_numpy(meta_data).float()
            # stastics
            # rel = (rWLs[names]).tolist()
            # pylab.hist(rel,20)
            # pylab.xlabel('Range')
            # pylab.ylabel('Count')
            # pylab.savefig('stat/{}.png'.format(design+'_golden'))
            # pylab.cla()
            # normalize
            size[:,0] = size[:,0]/(xh-xl)
            size[:,1] = size[:,1]/(yh-yl)
            pin_feature[:,0] = pin_feature[:,0]/(xh-xl)
            pin_feature[:,1] = pin_feature[:,1]/(yh-yl)
            # std
            rWLs = rWLs/(xh-xl+yh-yl)*2
            rWLs = rWLs/1000.0

            hpwls = hpwls/(xh-xl+yh-yl)*2
            hpwls = hpwls/1000.0

            orWLs = rWLs.copy()

            cell_size = torch.tensor(size, dtype=torch.float)
            edge_index = torch.tensor(incidence.T, dtype=torch.long)
            pins = torch.tensor(pin_feature,dtype=torch.float)
            gold = torch.tensor(golden,dtype=torch.float)
            #weight = torch.tensor(dist2macro,dtype=torch.float)
            #weight = 1/weight
            #summ = weight.sum(dim=-1).view(-1,1).repeat(1,int(len(macro_index)))
            #weight = weight/summ
            B = scatter(cell_size.new_ones(edge_index.size(1)),
                        edge_index[1], dim=0, dim_size=edge_index[1].max()+1, reduce='sum')

            B  = torch.index_select(B,dim=-1,index=edge_index[1]).clamp(0,50)
            
            #weight = torch.softmax(1/weight,dim=-1)
            # netlist is the same
            data = Data(x=cell_size, 
                    edge_index=edge_index,
                    pin_feature=pins,
                    macro_index=macro_index,
                    fixed_cell_index = fixed_cell_index)
            torch.save(data, osp.join(self.processed_dir, '{}.pt'.format(design)))
            for name in tqdm(names):
                if osp.exists(osp.join(self.processed_dir, 'data_{}.pt'.format(i))):
                    i += 1
                    continue
                if hpwls[name] == 0:
                    print('{}-{}'.format(design,name))
                pos_path = osp.join(pos_root,'%d.txt'%name)
                node_pos = torch.tensor(np.loadtxt(pos_path),dtype=torch.float) 

                node_pos[:,0] = (node_pos[:,0]-xl)/(xh-xl)
                node_pos[:,1] = (node_pos[:,1]-yl)/(yh-yl)

                macro_pos = node_pos[macro_index]

                #fake_pos = torch.matmul(weight,macro_pos)
                fake_pos = torch.zeros_like(node_pos)
                fake_pos[fixed_index] = node_pos[fixed_index]
        
                density = []

                ox = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                oy = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                for idx in macro_index:
                    pos = node_pos[idx]
                    size = cell_size[idx]
                    ox = torch.arange(0,1,self.bin_size,dtype=float).view(1,-1).repeat(self.num_bins,1)
                    oy = torch.arange(0,1,self.bin_size,dtype=float).view(-1,1).repeat(1,self.num_bins)

                    ox = torch.clamp((size[0]/2 + self.bin_size/2 - torch.abs(pos[0] - ox + size[0]/2 - self.bin_size/2)) / self.bin_size,0,1)
                    oy = torch.clamp((size[1]/2 + self.bin_size/2 - torch.abs(pos[1] - oy + size[1]/2 - self.bin_size/2)) / self.bin_size,0,1)

                    density.append((ox * oy).view(self.num_bins,self.num_bins,1))

                density = torch.cat(density,dim = -1)
                density_map = density.sum(dim=-1)

                pin_density = torch.zeros_like(density_map).view(-1)
                cnt_density = torch.zeros_like(density_map).view(-1)


                all_pin_pos = ((torch.index_select(node_pos,dim=0,index=edge_index[0]) + pins) /self.bin_size).long().clamp(0, self.num_bins - 1)
                def dd2d(index):
                    return index[:,1] * self.num_bins + index[:,0]
                pin_mask = torch.zeros(all_pin_pos.shape[0]).bool()
                for pidx in macro_index:
                    pin_mask |= (edge_index[0]==pidx)
                for pidx in fixed_cell_index:
                    pin_mask |= (edge_index[0]==pidx)    
                pin_pos = all_pin_pos[pin_mask] 
                indx = dd2d(pin_pos)

                pin_density = pin_density + scatter(all_pin_pos.new_ones(pin_pos.shape[0]), \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')
                    
                cnt_density = cnt_density + scatter(B[pin_mask], \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')

                pin_density /= pin_density.max()
                cnt_density /= cnt_density.max()

                pin_density = pin_density.view(self.num_bins,self.num_bins)
                cnt_density = cnt_density.view(self.num_bins,self.num_bins)

                pic = torch.cat([density_map.view(1,1,self.num_bins,self.num_bins),pin_density.view(1,1,self.num_bins,self.num_bins),cnt_density.view(1,1,self.num_bins,self.num_bins)],dim=1)

                y = torch.tensor(1.0,dtype=torch.long)
                if y.ndim <= 1:
                    y = y.view(1,-1)
                data = Data(x=torch.tensor(1,dtype=torch.float), 
                            edge_index=torch.tensor([1,1],dtype=torch.long),
                            y=y,
                            pin_feature=torch.tensor(1,dtype=torch.float),
                            macro_index=torch.tensor(1,dtype=torch.float),
                            fixed_cell_index = torch.tensor(1,dtype=torch.float),
                            golden=gold,
                            wl=torch.tensor(orWLs[name],dtype=torch.float),
                            #density = density.float(),
                            pic = pic.float(),
                            vias = torch.tensor(vias[name],dtype=torch.float),
                            short = torch.tensor(short[name],dtype=torch.float),
                            score = torch.tensor(score[name],dtype=torch.float),
                            meta_feature=meta_data,
                            cell_pos=macro_pos,
                            fake_pos=macro_pos,
                            hpwl=torch.tensor(hpwls[name],dtype=torch.float))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def balence_weight(self):
        wlist = []
        for label in self.labels:
            wt = {}
            weight = {}
            
            for design in self.raw_file_names:
                wt[design] = []
            id2design = lambda idx : [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]
            for i, data in enumerate(self):
                y = getattr(data,label)
                design = id2design(i)
                if design in self.raw_file_names:
                    wt[design].append(y)
            for design in self.raw_file_names:
                weight[design] = 1. / np.var(wt[design])
                wlist.append(weight[design])
            self.weight[label] = weight
        scaler = 1. / np.mean(wlist)
        for design in self.raw_file_names:
            for label in self.labels:
                self.weight[label][design] *= scaler
        return weight



    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        data.short = data.short/100000
        data.score = data.score/100000000
        data.vias = data.vias/100000
        data.design = [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]

        data.x=self.netlist[data.design].x
        data.edge_index=self.netlist[data.design].edge_index
        data.pin_feature=self.netlist[data.design].pin_feature
        data.macro_index=self.netlist[data.design].macro_index
        data.fixed_cell_index = self.netlist[data.design].fixed_cell_index
        data.all = torch.cat([data.wl.view(1,1),data.vias.view(1,1),data.short.view(1,1)],dim=1)
        return data


class PlainClusterSet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, classes=20, test_files=['mgc_fft_a','mgc_matrix_mult_b'], train_files=['mgc_fft_b'], device='cpu'):
        self.tot_file_num = None # int
        self.file_num = None # dict, file nums for each design
        self.ptr = None
        self.classes = classes
        self.num_bins = 224
        self.bin_size = 1./224
        self.label_key = None
        self.netlist = {}
        self.data = []
        self.labels = ['wl','vias','short']
        self.weight = {}
        self.train_file_names = train_files
        self.test_file_names = test_files
        self.device = device
        super(PlainClusterSet, self).__init__(root, transform, pre_transform)
        for design in self.raw_file_names:
            self.netlist[design] = torch.load(osp.join(self.processed_dir, '{}.pt'.format(design))).to(device)
        for i in range(len(self.processed_file_names)):
            self.data.append(self.pre_load_data(i).to(device))
        #self.balence_weight()

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_cluster')

    @property
    def raw_file_names(self):
        names_path = osp.join(self.root,'raw','all.names')
        names = np.loadtxt(names_path,dtype=str)
        if names.ndim == 0:
            return [str(names)]
        return names.tolist()
    
    @property
    def num_features(self):
        return self[0].x.size(1) + self[0].cell_pos.size(1) + self[0].pin_feature.size(1) + 1 # macro label

    @property
    def num_classes(self):
        return int(self.classes)

    @property
    def processed_file_names(self):
        if self.tot_file_num is None:
            self.tot_file_num = 0
            self.file_num = {}
            self.ptr = {}
            for design in self.raw_file_names:
                path = osp.join(self.raw_dir,design)
                name_path = osp.join(path,'names.txt')
                names = np.array(pd.read_table(name_path,header=None)).reshape(-1)
                self.tot_file_num += names.shape[0]
                self.file_num[design] = names.shape[0]
            self.ptr[self.raw_file_names[0]] = 0
            for i in range(1,int(len(self.raw_file_names))):
                self.ptr[self.raw_file_names[i]] = self.ptr[self.raw_file_names[i-1]] + self.file_num[self.raw_file_names[i-1]]
        return ['data_%d.pt'%i for i in range(0,self.tot_file_num)]


    def process(self):

        i = 0
        for design in self.raw_file_names:
            # paths
            path = osp.join(self.raw_dir,design)
            size_path = osp.join(path,'node_size.txt')
            name_path = osp.join(path,'names.txt')
            pos_root = osp.join(path,'node_pos')
            pin_path = osp.join(path,'pins.txt')
            region_path = osp.join(path,'region.txt')
            macro_path = osp.join(path,'macro_index.txt')
            hpwl_path = osp.join(path,'hpwl.txt')
            meta_path = osp.join(path,'meta.txt')
            label_path = osp.join(path,'labels.txt')
            hedge_w_path = osp.join(path, 'edge_weights.txt')
            # loading ...
            pins = np.loadtxt(pin_path)
            size = np.loadtxt(size_path)
            hedge_w = torch.tensor(np.load(hedge_w_path))
            
            incidence = pins[:,:2]
            pin_feature = pins[:,2:]
            xl,yl,xh,yh = np.loadtxt(region_path)
            
            macro_index = torch.tensor(np.loadtxt(macro_path),dtype=torch.long)
            names = np.loadtxt(name_path,dtype=int)

            hpwls = np.loadtxt(hpwl_path)
            meta_data = np.loadtxt(meta_path)
            labels = np.loadtxt(label_path)
            fixed_index = macro_index#torch.from_numpy(np.loadtxt(fixed_path)).long()
            fixed_cell_index = torch.tensor([i for i in fixed_index if i not in macro_index]).long()
            
            rWLs = labels[:,0]
            vias = labels[:,1]
            short = labels[:,2]
            score = labels[:,3]

            meta_data[5] = meta_data[5]/(yh-yl)
            meta_data[8] = meta_data[8]/(yh-yl)/(xh-xl)
            meta_data[9] = meta_data[9]/(yh-yl)/(xh-xl)
            meta_data[10] = meta_data[10]/(yh-yl)/(xh-xl)

            meta_data = torch.from_numpy(meta_data).float()
            size[:,0] = size[:,0]/(xh-xl)
            size[:,1] = size[:,1]/(yh-yl)
            pin_feature[:,0] = pin_feature[:,0]/(xh-xl)
            pin_feature[:,1] = pin_feature[:,1]/(yh-yl)
            # std
            rWLs = rWLs/(xh-xl+yh-yl)*2
            rWLs = rWLs/1000.0
            hpwls = hpwls/(xh-xl+yh-yl)*2
            hpwls = hpwls/1000.0

            cell_size = torch.tensor(size, dtype=torch.float)
            edge_index = torch.tensor(incidence.T, dtype=torch.long)
            pins = torch.tensor(pin_feature,dtype=torch.float)

            B = scatter(cell_size.new_ones(edge_index.size(1)),
                        edge_index[1], dim=0, dim_size=edge_index[1].max()+1, reduce='sum')

            B  = torch.index_select(B,dim=-1,index=edge_index[1]).clamp(0,50)
            
            degree = scatter(cell_size.new_ones(edge_index.size(1)), edge_index[0], dim=0, )


            hedge_w = torch.index_select(hedge_w, dim=-1, index=edge_index[1])
            #weight = torch.softmax(1/weight,dim=-1)
            # netlist is the same
            data = Data(
                    x=cell_size, # x = [size[2 or 16], degree[1], pins[1]]
                    edge_index=edge_index,
                    edge_weight=hedge_w,
                    pin_feature=pins,
                    macro_index=macro_index)
            torch.save(data, osp.join(self.processed_dir, '{}.pt'.format(design)))
            for name in tqdm(names):
                if osp.exists(osp.join(self.processed_dir, 'data_{}.pt'.format(i))):
                    i += 1
                    continue
                if hpwls[name] == 0:
                    print('{}-{}'.format(design,name))
                pos_path = osp.join(pos_root,'%d.txt'%name)
                node_pos = torch.tensor(np.loadtxt(pos_path),dtype=torch.float) 

                node_pos[:,0] = (node_pos[:,0]-xl)/(xh-xl)
                node_pos[:,1] = (node_pos[:,1]-yl)/(yh-yl)
                #fake_pos = torch.matmul(weight,macro_pos)
                fake_pos = torch.zeros_like(node_pos)
                fake_pos[fixed_index] = node_pos[fixed_index]
        
                density = []

                ox = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                oy = macro_index.new_zeros(self.num_bins,self.num_bins).float()
                for idx in macro_index:
                    pos = node_pos[idx]
                    size = cell_size[idx]
                    ox = torch.arange(0,1,self.bin_size,dtype=float).view(1,-1).repeat(self.num_bins,1)
                    oy = torch.arange(0,1,self.bin_size,dtype=float).view(-1,1).repeat(1,self.num_bins)

                    ox = torch.clamp((size[0]/2 + self.bin_size/2 - torch.abs(pos[0] - ox + size[0]/2 - self.bin_size/2)) / self.bin_size,0,1)
                    oy = torch.clamp((size[1]/2 + self.bin_size/2 - torch.abs(pos[1] - oy + size[1]/2 - self.bin_size/2)) / self.bin_size,0,1)

                    density.append((ox * oy).view(self.num_bins,self.num_bins,1))

                density = torch.cat(density,dim = -1)
                density_map = density.sum(dim=-1)

                pin_density = torch.zeros_like(density_map).view(-1)
                cnt_density = torch.zeros_like(density_map).view(-1)


                all_pin_pos = ((torch.index_select(node_pos,dim=0,index=edge_index[0]) + pins) /self.bin_size).long().clamp(0, self.num_bins - 1)
                def dd2d(index):
                    return index[:,1] * self.num_bins + index[:,0]
                pin_mask = torch.zeros(all_pin_pos.shape[0]).bool()
                for pidx in macro_index:
                    pin_mask |= (edge_index[0]==pidx)
                for pidx in fixed_cell_index:
                    pin_mask |= (edge_index[0]==pidx)    
                pin_pos = all_pin_pos[pin_mask] 
                indx = dd2d(pin_pos)

                pin_density = pin_density + scatter(all_pin_pos.new_ones(pin_pos.shape[0]), \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')
                    
                cnt_density = cnt_density + scatter(B[pin_mask], \
                        indx, dim=0, dim_size=self.num_bins * self.num_bins, reduce='sum')

                pin_density /= pin_density.max()
                cnt_density /= cnt_density.max()

                pin_density = pin_density.view(self.num_bins,self.num_bins)
                cnt_density = cnt_density.view(self.num_bins,self.num_bins)

                pic = torch.cat([density_map.view(1,1,self.num_bins,self.num_bins),pin_density.view(1,1,self.num_bins,self.num_bins),cnt_density.view(1,1,self.num_bins,self.num_bins)],dim=1)

                y = torch.tensor(1.0,dtype=torch.long)
                if y.ndim <= 1:
                    y = y.view(1,-1)
                data = Data(# position[ll][2 or 16]
                            x=fake_pos.float(),
                            # ...
                            edge_index=torch.tensor([1,1],dtype=torch.long),
                            # label = [hpwl, rwl, vias, short, score]
                            y=torch.tensor([hpwls[name], rWLs[name], vias[name], short[name], score[name]],dtype=torch.float), 
                            # density_map
                            pic = pic.float())

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

    def balence_weight(self):
        wlist = []
        for label in self.labels:
            wt = {}
            weight = {}
            
            for design in self.raw_file_names:
                wt[design] = []
            id2design = lambda idx : [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]
            for i, data in enumerate(self):
                y = getattr(data,label)
                design = id2design(i)
                if design in self.raw_file_names:
                    wt[design].append(y)
            for design in self.raw_file_names:
                weight[design] = 1. / np.var(wt[design])
                wlist.append(weight[design])
            self.weight[label] = weight
        scaler = 1. / np.mean(wlist)
        for design in self.raw_file_names:
            for label in self.labels:
                self.weight[label][design] *= scaler
        return weight



    def len(self):
        return len(self.processed_file_names)

    def pre_load_data(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        data.short = data.short/100000
        data.score = data.score/100000000
        data.vias = data.vias/100000
        data.design = [design for design in self.raw_file_names if idx >= self.ptr[design] and  idx < self.ptr[design] + self.file_num[design]][0]

        data.x=self.netlist[data.design].x
        data.edge_index=self.netlist[data.design].edge_index
        data.pin_feature=self.netlist[data.design].pin_feature
        data.macro_index=self.netlist[data.design].macro_index
        data.fixed_cell_index = self.netlist[data.design].fixed_cell_index
        data.all = torch.cat([data.wl.view(1,1),data.vias.view(1,1),data.short.view(1,1)],dim=1)
        return data

    def get(self, idx):
        return self.data[idx]

if __name__=='__main__':
    Set = ClusterSet('data')
    pic = Set[1000].pic.cpu().view(-1,224,224).numpy()
    pic = np.array(pic[2] * 255,dtype=np.uint8)#.transpose(1, 2, 0)
    im = Image.fromarray(pic)
    #im.convert('L')
    im.save('test.jpg') 
    exit(0)

    


