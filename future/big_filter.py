#import gluonbook as gb
from mxnet import autograd,nd,init,gluon
from mxnet.gluon import loss as gloss,data as gdata,nn,utils as gutils
import mxnet as mx
net = nn.Sequential()

with net.name_scope():
    net.add(
        nn.Conv2D(channels=32, kernel_size=3, activation='relu'),
        nn.Conv2D(channels=32, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation='sigmoid'),
        nn.Dense(10, activation='sigmoid')
    )

lr = 0.5
batch_size=256
ctx = mx.gpu()
net.initialize(init=init.Xavier(), ctx=ctx)

train_data, test_data = gb.load_data_fashion_mnist(batch_size)
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate' : lr})
loss = gloss.SoftmaxCrossEntropyLoss()
num_epochs = 30

def train(train_data, test_data, net, loss, trainer,num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for x,y in train_data:
            with autograd.record():
                x = x.as_in_context(ctx)
                y = y.as_in_context(ctx)
                y_hat=net(x)
                l = loss(y_hat,y)
            l.backward()
            total_loss += l
            trainer.step(batch_size)
        mx.nd.waitall()
        print("Epoch [{}]: Loss {}".format(epoch, total_loss.sum().asnumpy()[0]/(batch_size*len(train_data))))

if __name__ == '__main__':
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    ctx
    gb.train(train_data,test_data,net,loss,trainer,ctx,num_epochs)