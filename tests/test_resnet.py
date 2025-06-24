import sys

sys.path.append("./")
from agent.networks.resnet_encoder import ResNet18Encoder

if __name__ == "__main__":
    # resnet = torchvision.models.resnet18()
    # import ipdb; ipdb.set_trace()
    # print((list(resnet.children())[-3]))
    # for c in resnet.children():
    #     print(c)

    resnet = ResNet18Encoder(256)
