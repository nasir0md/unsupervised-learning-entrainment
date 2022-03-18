# from aeent import *
from entrainment_config import *
from ecdc import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

model_name = 'models/trained_VAE_nonorm_nopre_l1.pt'


SEED=448

#------------------------------------------------------------------
#Uncomment for parsing inputs

parser = argparse.ArgumentParser(description='entrainment testing')
parser.add_argument('--no-cuda', action='store_true', default=False,
	help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
	help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



hff = h5py.File('data/test_Fisher_nonorm_nopre.h5', 'r')
X_test = np.array(hff['dataset'])




model = VAE().double()



model = torch.load(model_name)
model.eval()
if args.cuda:
    model.cuda()



if 'l1' in model_name:
    p=1
elif 'l2' in model_name:
    p=2
else:
    print("need better model name")
    p=2

    pdb
test_loss = 0
fake_test_loss = 0
Loss=[]
Fake_loss = []
# for batch_idx, (x_data, y_data) in enumerate(test_loader):
for idx, data in enumerate(X_test):
    x_data = data[:228]
    y_data = data[228:-1]
    idx_same_spk =list(np.where(X_test[:,-1]==data[-1]))[0]

    ll = random.choice(list(set(idx_same_spk) -set([idx])))
    spk = int(data[-1])

    x_data = Variable(torch.from_numpy(x_data))
    y_data = Variable(torch.from_numpy(y_data))

    y_fake_data = X_test[ll,228:-1]

    y_fake_data = Variable(torch.from_numpy(y_fake_data))

    if args.cuda:
        x_data = x_data.cuda()
        y_data = y_data.cuda()
        y_fake_data = y_fake_data.cuda()

    recon_batch = model(x_data)


    z_x = model.embedding(x_data)
    z_y = model.embedding(y_data)
    # z_x = x_data
    # z_y = y_data
    loss_real = lp_distance(z_x, z_y, p).data[0]
    # loss_real = loss_function(z_x, z_y, mu, logvar)
    

    z_y_fake = model.embedding(y_fake_data)
    # z_y_fake = y_fake_data

    loss_fake = lp_distance(z_x, z_y_fake, p).data[0]
    # loss_fake = loss_function(z_x, z_y_fake, mu, logvar)

    test_loss += loss_real
    fake_test_loss += loss_fake 

    Loss.append(loss_real)
    Fake_loss.append(loss_fake)
    # print loss_real, loss_fake



test_loss /= X_test.shape[0]

fake_test_loss /= X_test.shape[0]

Loss=np.array(Loss)
Fake_loss=np.array(Fake_loss)
print("Total Real Loss:"+str(test_loss) + "Total Fake Loss:" + str(fake_test_loss))

print(float(np.sum(Loss < Fake_loss))/Loss.shape[0])




