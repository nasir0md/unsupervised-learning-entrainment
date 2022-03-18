from entrainment_config import *

model_path = model_path
work_dir = work_dir
model_name = model_path + 'triplet_64d_50ep_fisher.pkl'

SEED = 448
# --------------------------------------------------------------------------
# Load model

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(SEED)


embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)

model.load_state_dict(torch.load(model_name))
# model = torch.load(model_name)

model.eval()

if cuda:
    model.cuda()

def lp_distance(x1, x2, p):
    dist = torch.dist(x1, x2,p)
    return dist

# --------------------------------------------------------------------------
# Load data
hff = h5py.File(work_dir + 'data/test_Fisher_nonorm_nopre.h5', 'r')
X_test = np.array(hff['dataset'])
p=2

results = []
for k in range(20):

    Loss=[]
    Fake_loss = []
    # for batch_idx, (x_data, y_data) in enumerate(test_loader):
    N = int(X_test[-1,-1])

    for spk_pair in range(1,N+1):

        idx_same_spk =list(np.where(X_test[:,-1]==spk_pair))[0]


        test_loss = 0
        fake_test_loss = 0

        for idx in idx_same_spk:

            ll = random.choice(list(set(idx_same_spk) -set([idx])))
            x_data = X_test[idx,:228]
            y_data = X_test[idx,228:-1]
            y_fake_data = X_test[ll,228:-1]

            x_data = Variable(torch.from_numpy(x_data)).double()
            y_data = Variable(torch.from_numpy(y_data)).double()
            y_fake_data = Variable(torch.from_numpy(y_fake_data)).double()

            if cuda:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
                y_fake_data = y_fake_data.cuda()

            z_x = model.get_embedding(x_data)
            z_y = model.get_embedding(y_data)
            z_y_fake = model.get_embedding(y_fake_data)

            loss_real = lp_distance(z_x, z_y, p).data        
            loss_fake = lp_distance(z_x, z_y_fake, p).data
            test_loss += loss_real
            fake_test_loss += loss_fake 
        # pdb.set_trace()
        Loss.append(test_loss)
        Fake_loss.append(fake_test_loss)
        # print loss_real, loss_fake

    Loss=np.array(Loss)
    Fake_loss=np.array(Fake_loss)

    total_test_loss = np.sum(Loss)/Loss.shape[0]

    total_fake_test_loss = np.sum(Fake_loss)/Loss.shape[0]


    print(("Total Real Loss:"+str(total_test_loss) + "Total Fake Loss:" + str(total_fake_test_loss)))

    print((float(np.sum(Loss < Fake_loss))/Loss.shape[0]))

    results.append(float(np.sum(Loss < Fake_loss))/Loss.shape[0])


print((np.mean(np.array(results))))