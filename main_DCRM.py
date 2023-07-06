import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn.functional as F
from skimage import color
import scipy.io as sio
import numpy as np
from network_w import *
import os
import pickle
import IntegrationLoss
from scipy.interpolate import griddata
torch.manual_seed(2022)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# New parameters
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 500

n_epochs = 40000

display_step = 1000
save_step = 2000
batch_size = 4
lr = 0.0001
beta_1 = 0.5
beta_2 = 0.999
target_shape = 128
c_lambda = 10
num_channels = 250

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#    print("Running on the GPU")
#else:
device = torch.device("cpu")
#    print("Running on the CPU")

input_dim = 2
real_dim = 1

prefix = './Data_BC_250'


u3val = np.array([sio.loadmat(os.path.join(prefix, "BCMultiPoissonCalc_" + str(i) + ".mat"))['u'] for i in range(1,num_channels+1)])
poisson_f = np.array([sio.loadmat(os.path.join(prefix, "BCMultiPoissonCalc_" + str(i) + ".mat"))['gf'] for i in range(1,num_channels+1)])
inputs = torch.tensor(np.expand_dims(poisson_f, axis=1))
u3val = torch.tensor(np.expand_dims(u3val, axis=1))
BCval = torch.zeros_like(u3val)
for i in range(u3val.shape[0]):
    BCval[i, 0, 1:127, 1:127] = torch.zeros((126,126))
    BCNP = BCval[i, 0, :, :].detach().numpy()

true_sol = torch.as_tensor(u3val.reshape((u3val.shape[0],1,u3val.shape[2], u3val.shape[3])))



num_channelsTest = 1000
prefixTest = './Data_BC_1000'

u3valTest = np.array([sio.loadmat(os.path.join(prefixTest, "BCMultiPoissonCalc_" + str(i) + ".mat"))['u'] for i in range(1,num_channelsTest+1)])
poisson_fTest = np.array([sio.loadmat(os.path.join(prefixTest, "BCMultiPoissonCalc_" + str(i) + ".mat"))['gf'] for i in range(1,num_channelsTest+1)])
inputsTest = torch.tensor(np.expand_dims(poisson_fTest , axis=1))
u3valTest = torch.tensor(np.expand_dims(u3valTest , axis=1))
BCvalTest = torch.zeros_like(u3valTest)
for i in range(u3valTest .shape[0]):

    BCvalTest[i,0,:,:] = u3valTest[i,0,:,:]
    BCvalTest[i, 0, 1:127, 1:127] = torch.zeros((126,126))

true_solTest  = torch.as_tensor(u3valTest.reshape((u3valTest .shape[0],1,u3valTest.shape[2], u3valTest.shape[3])))




xxxx130 = np.linspace(0, 1, 130)
delx130 = np.abs(xxxx130[1] - xxxx130[0])

xxxx = np.linspace(0,1,128)
delx = np.abs(xxxx[1] - xxxx[0])

xxxx126 = np.linspace(0,1,126)
delx126 = np.abs(xxxx126[1] - xxxx126[0])

intLoss = IntegrationLoss.IntegrationLoss('trapezoidal', 2)


x = torch.linspace(0, 1, 128 )
y = torch.linspace(0, 1, 128 )
rx, ry = torch.meshgrid(x, y)
rx = rx.to(device)
ry = ry.to(device)
rxd = rx.cpu().detach().numpy()
ryd = ry.cpu().detach().numpy()


idx_plot = [0,50,100]
for i in range(len(idx_plot)):

    psm = plt.pcolormesh(rxd,ryd,u3val[idx_plot[i],0, :, :].detach().numpy())
    plt.colorbar(psm)
    ST = 'Images/TrueOutput_'+str(i)+'.png'
    plt.savefig(ST)
    plt.close('all')


gen = UNet(input_dim, real_dim).to(device)


gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_opt, 300 * 2500)
scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=4000, gamma=0.8)

gen = gen.apply(weights_init)


stencildudx =np.array(((-3.0,4. , -1.0)))
stencildudy =np.array(((-3.0),(4.) ,(-1.0)))

stencildudx_2 =np.array(((-1.0,4. , -3.0)))
stencildudy_2 =np.array(((-1.0),(4.) ,(-3.0)))






def du_dx(x, index, input_dim, real_dim):
    m = torch.nn.Conv2d(1, 1, (3,1), stride=(1,1), groups=real_dim).to(device)

    if index == 1:
        for i in range(3):
            for k in range(1):
                m.weight.data[k, 0, i, 0] = stencildudx[i]
    else:
        for i in range(3):
            for k in range(1):
                m.weight.data[k, 0, i, 0] = stencildudx_2[i]
    m.bias.data = torch.zeros((1))

    with torch.autograd.no_grad():
        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)

    m= m.to(device)
    return m





def du_dy(x, index, input_dim, real_dim):
    m = torch.nn.Conv2d(1, 1, (1,3), stride=1, groups=real_dim).to(device)

    if index == 1:
        for j in range(3):
            for k in range(1):
                m.weight.data[k, 0, 0, j] = stencildudy[j]
    else:
        for j in range(3):
            for k in range(1):
                m.weight.data[k, 0, 0, j] = stencildudy_2[j]

    m.bias.data = torch.zeros((1))
    with torch.autograd.no_grad():

        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)

    m= m.to(device)
    return m


def getGenOutputPadding(x):
    fake = gen(x)
    n = x.shape[2]
    m = x.shape[3]
    xx = torch.zeros((x.shape[0], 1, n + 2, m + 2)).to(device)

    xx[:, :, 1:n + 1, 1:m + 1] = fake

    xx[:, 0, 0, 1:m + 1] = x[:,1,0, 0:m]
    xx[:, 0, n + 1, 1:m + 1] = x[:,1,n-1, 0:m]
    xx[:, 0, 1:n + 1, 0] = x[:,1,0:n, 0]
    xx[:, 0, 1:n + 1, m + 1] = x[:,1,0:n, m-1]


    xx[:, 0, 0, m + 1] = 0.5 * (xx[:, 0, 0, m] + xx[:, 0, 1, m + 1])
    xx[:, 0, n + 1, 0] = 0.5 * (xx[:, 0, n, 0] + xx[:, 0, n + 1, 1])
    xx[:, 0, 0, 0] = 0.5 * (xx[:, 0, 0, 1] + xx[:, 0, 1, 0])
    xx[:, 0, n + 1, m + 1] = 0.5 * (xx[:, 0, n + 1, m] + xx[:, 0, n, m + 1])


    return xx



def interpo(condition_st):
    n = 128
    linx128 = np.linspace(0, 1, n)

    linx130 = np.linspace(0, 1, n + 2)
    z1_out = np.zeros(((condition_st.shape[0], 1, 130, 130)))
    for mm in range(condition_st.shape[0]):
        val = condition_st[mm, 0, :, :].detach().cpu().numpy()
        xv_128, yv_128 = np.meshgrid(linx128, linx128, indexing='ij')
        xv_130, yv_130 = np.meshgrid(linx130, linx130, indexing='ij')

        points = np.zeros(( 128 * 128, 2))
        values = np.zeros(( 128 * 128, 1))
        iter = 0
        for i in range(128):
            for j in range(128):
                points[iter, 0] = xv_128[i, j]
                points[iter, 1] = yv_128[i, j]
                values[iter, 0] = val[i, j]
                iter = iter + 1

        grid_z1 = griddata(points, values[:, 0], (xv_130, yv_130), method='linear')
        z1_out[mm,0,:,:] = grid_z1.reshape((1, 1, 130, 130))

    return torch.as_tensor(z1_out)


# train
def train(save_model=False):


    condition = inputs.to(device).float()
    bc = BCval.to(device).float()
    inpComb = torch.cat((condition,bc),1)


    conditionTest = inputsTest.to(device).float()
    bcTest = BCvalTest.to(device).float()
    inpCombTest = torch.cat((conditionTest, bcTest), 1)


    inter_condition = interpo(condition)
    inter_condition = inter_condition.to(device).float()

    n = condition.shape[2]
    m = condition.shape[3]
    x = torch.linspace(0, 1, n + 2)
    y = torch.linspace(0, 1, m + 2)
    rx, ry = torch.meshgrid(x, y)
    rx = rx.to(device)
    ry = ry.to(device)
    rxd = rx.cpu().detach().numpy()
    ryd = ry.cpu().detach().numpy()


    W = torch.zeros(rx.shape).to(device)
    W = W.reshape(1, 1, rx.shape[0], rx.shape[1])


    convGraddudx = du_dx(W, 1, input_dim, real_dim)
    convGraddudy = du_dy(W, 1, input_dim, real_dim)

    convGraddudx2 = du_dx(W, 2, input_dim, real_dim)
    convGraddudy2 = du_dy(W, 2, input_dim, real_dim)

    inter_true = interpo(true_sol)
    inter_true = inter_true.to(device).float()
    inter_true_np = inter_true.cpu().detach().numpy()

    inter_trueTest = interpo(true_solTest)
    inter_trueTest = inter_trueTest.to(device).float()


    torch_dataset = TensorDataset(inpComb, inter_true,inter_condition)
    dataloader = DataLoader(dataset=torch_dataset, batch_size=2, shuffle=False)


    torch_datasetTest = TensorDataset(inpCombTest, inter_trueTest)
    dataloaderTest = DataLoader(dataset=torch_datasetTest, batch_size=2, shuffle=False)

    cur_step = 0
    display_step = 500
    plot_step = 5000

    Losses_list = []

    while cur_step < 600000:
        meanLossCompare = 0.0
        meanLoss = 0.0

        for inpt, outpt, interOut in tqdm(dataloader):

            out = getGenOutputPadding(inpt)


            out_dudx = (1 / (2 * delx130)) * convGraddudx(out)
            out_dudy = (1 / (2 * delx130)) * convGraddudy(out)

            out_dudx2 = (1 / (2 * delx130)) * convGraddudx2(out)
            out_dudy2 = (1 / (2 * delx130)) * convGraddudy2(out)

            I_in1 = torch.pow(out_dudx, 2)
            I_in12 = torch.pow(out_dudx2, 2)
            I_in2 = torch.pow(out_dudy, 2)
            I_in22 = torch.pow(out_dudy2, 2)
            internal1 = intLoss.lossInternalEnergy(0.5 * (I_in1[:, : :, :] + I_in12[:, : :, :]), dx=delx, dy=delx130,
                                                   shape=I_in1[:, : :, :].shape)
            internal2 = intLoss.lossInternalEnergy(0.5 * (I_in2[:, : :, :] + I_in22[:, : :, :]), dx=delx130, dy=delx,
                                                   shape=I_in2[:, : :, :].shape)


            f_u = out * (interOut)


            internal_appl_f = intLoss.lossInternalEnergy(f_u[:, :, :, :], dx=delx130, dy=delx130,
                                                         shape=f_u[:, :, :, :].shape)

            internal = internal1 + internal2
            loss = 0.5 * internal - internal_appl_f
            gen_opt.zero_grad()

            loss.backward()
            gen_opt.step()

            meanLoss += loss
            meanLossCompare = meanLossCompare+ torch.sum(torch.abs(out - outpt))
            if cur_step % display_step == 0:

                with torch.no_grad():
                    lossTest = torch.tensor(0.0, requires_grad=False)
                    for inptTest, outptTest in tqdm(dataloaderTest):
                        outTest = getGenOutputPadding(inptTest)
                        lossTest = lossTest+ torch.sum(torch.abs(outTest - outptTest))



                meanLoss = meanLoss / display_step
                meanLossCompare = meanLossCompare/ display_step

                Losses_list.append([cur_step, meanLoss.item(), meanLossCompare.item(),lossTest.item()])

                with open('Images/energyLosses.pickle', 'wb') as handle:
                    pickle.dump(Losses_list, handle)

                print(
                    f"Step {cur_step}: Loss {meanLoss}: loss_com: {meanLossCompare}: loss_test: {lossTest}")


            if cur_step % plot_step == 0:
                with torch.no_grad():
                    out_test = getGenOutputPadding(inpComb[idx_plot,:,:,:])

                outnp = out_test.cpu().detach().numpy()

                for oo in range(len(idx_plot)):
                    ST = 'Images/prediction' + str(cur_step) +'_'+str(oo)+ '.png'
                    psm = plt.pcolormesh(rxd, ryd, outnp[oo, 0, :,:])
                    plt.colorbar(psm)
                    plt.savefig(ST)
                    plt.close('all')

                    ST = 'Images/AbsError' + str(cur_step) +'_'+str(oo)+ '.png'
                    psm = plt.pcolormesh(rxd, ryd, np.abs(outnp[oo, 0, :, :] - inter_true_np[idx_plot[oo], 0, :, :]))
                    plt.colorbar(psm)
                    plt.savefig(ST)
                    plt.close('all')
            if cur_step % save_step== 0:
                ST = 'Images/EnergyModel' + '_'+ '.pt'
                gen.saveModel(ST)
                print('Saved:  ', ST)

            cur_step = cur_step+1

train(save_model=True)

