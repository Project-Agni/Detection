import numpy as np
import torch
import torch.nn.functional as F

from agents.cnn import CNN

decay_rate = 0.99
gamma = 0
epsilon = -0.001


class QNet(CNN):
    def __init__(self):
        super(QNet, self).__init__()

    @classmethod
    def train_model(
        cls, args, model, device, train_loader, optimizer, epoch, env=None, **kwargs
    ):
        model.train()
        _ = env.reset()
        batch_idx, (data, target) = next(enumerate(train_loader))
        batch_idx = -1

        for next_batch_idx, (next_data, next_target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            next_data, next_target = next_data.to(device), next_target.to(device)
            env.save_step_targets(target)
            optimizer.zero_grad()

            q, action = get_action(data, model, env)
            q = q.to(device)
            _, rewards, _, _ = env.step(action)

            q_prime = model(next_data)
            indx = torch.argmax(q_prime, dim=1)
            # sx = np.arange(len(indx))
            targets = rewards + gamma * indx
            # q[sx, action] = targets

            # output = model(data)
            loss = F.nll_loss(q, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                if args.dry_run:
                    break

            batch_idx, (data, target) = next_batch_idx, (next_data, next_target)

    @classmethod
    def train_agent(cls, online_net, target_net, optimizer, batch, **kwargs):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state):
        qvalue = self.forward(state)
        actions = torch.argmax(qvalue, dim=1)
        return qvalue, actions


def get_action(state, target_net, env):
    if np.random.rand() <= epsilon:
        bs = len(state)
        return torch.zeros([bs, 2], dtype=torch.float), env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())
