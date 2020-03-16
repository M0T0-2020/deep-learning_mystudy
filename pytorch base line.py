pretrain_path = Path('../input/pytorch-kernel-submission/')
compe_data_path = Path('../input/aptos2019-blindness-detection/')

change_trn_val = True

if not change_trn_val:
    trn_df = pd.read_csv(pretrain_path/'trn_df.csv')
    val_df = pd.read_csv(pretrain_path/'val_df.csv')
    trn_df.to_csv('trn_df.csv', index=False)
    val_df.to_csv('val_df.csv', index=False)
else:
    train_df = pd.read_csv(compe_data_path/'train.csv')
    test_df = pd.read_csv(compe_data_path/'sample_submission.csv')
    d1 = train_df[train_df.diagnosis==0].sample(frac=0.2)
    d2 = train_df[train_df.diagnosis==2].sample(frac=0.4)
    d134 = train_df[(train_df.diagnosis==1) | (train_df.diagnosis==3)|(train_df.diagnosis==4)]
    df = pd.concat([d1,d2])
    df = pd.concat([df, d134])
    trn_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    trn_df.to_csv('trn_df.csv', index=False)
    val_df.to_csv('val_df.csv', index=False)


def make_labels(train_df, test_df=False):
    if test_df==True:
        c = len(train_df)
        r = 5
        l = np.zeros((c, r), dtype=int)
        #print(l)
        train_df = pd.concat([train_df, pd.DataFrame(l, columns=['0','1','2','3','4'])], axis=1)
        train_df = train_df.drop('diagnosis', axis=1)
        print(train_df.head(3))
        return train_df
    else:
        labels = pd.get_dummies(train_df['diagnosis'])
        train_df = pd.concat([train_df, labels], axis=1).drop('diagnosis', axis=1)
        print(train_df.head(3))
        return train_df
    
trn_df = make_labels(trn_df)
val_df = make_labels(val_df)

SEED = 123

def train_model(trn_df, val_df):
    num_epochs = 40
    batch_size = 60
    test_batch_size = 100
    lr = 1e-4
    eta_min = 1e-5
    t_max = 10
    numclass = 5
    
    print('train_test_split...')
    ROOT_DIR = compe_data_path/'train_images'
    
    trn = MyDataset(trn_df, ROOT_DIR, transform=transforms.Compose([transforms.RandomChoice([
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                transforms.RandomRotation(degrees=45, expand=True)]),
                                                                    
                                                                transforms.RandomChoice([
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                transforms.RandomRotation(degrees=45, expand=True)]),
                                                                    
                                                                transforms.Resize((150, 150)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])]))
    
    val = MyDataset(val_df, ROOT_DIR, transform=transforms.Compose([transforms.RandomChoice([
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                transforms.RandomRotation(degrees=45, expand=True)]),
                                                                
                                                                transforms.RandomChoice([
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                transforms.RandomRotation(degrees=45, expand=True)]),
                                                                    
                                                                transforms.Resize((150, 150)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])]))

    print('load_data...')
    train_loader = torch.utils.data.DataLoader(trn, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val, batch_size=test_batch_size, shuffle=True)
        
    
    model = Classifier()
    model.load_state_dict(torch.load(pretrain_path/"net.pt"))
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


    best_epoch = -1

    for epoch in range(num_epochs):
        print('epoch', epoch)
        start_time = time.time()
        # change model to be train_mode 
        model.train()
        avg_loss = 0.

#         for x_batch, y_batch in progress_bar(train_loader, parent=mb):
        for x_batch, y_batch in tqdm(train_loader):
            preds = model(x_batch.cuda())
            loss = criterion(preds.squeeze(1), y_batch.cuda())
            #print(loss)
            #print('train')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
            #print(avg_loss)

            
        # change model to be validation_mode
        model.eval()
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(test_loader):
            #print('test')
            
            preds = model(x_batch.cuda()).detach()
            loss = criterion(preds.squeeze(1), y_batch.cuda())

            avg_val_loss += loss.item() / len(test_loader)

        if (epoch + 1) % 1 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s')
    
    
    torch.save(model.state_dict(), 'net.pt')
    
    return {
        'best_epoch': best_epoch
    }
