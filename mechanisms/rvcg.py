import torch
import argparse
from utility.efficient_allocation import delete_agent, get_v, get_v_sum_but_i, oracle
def parse_cli_args():
    """ parse commandline arguments
    """
    # ====== training settings ======
    parser = argparse.ArgumentParser()
    # optimization setting
    
    parser.add_argument('-a', type=int,   default=2)
    parser.add_argument('-i', type=int,   default=2)

    cfg = parser.parse_args()
    return cfg

def VCG(batch, language="marginal"):
    """
    VCG efficient and truthful mechanism.
    :param batch: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :return: VCG prices t, tensor shaped as (batch_size, n_agents)
    """
    allocation = oracle(batch, language)
    v = get_v(batch, allocation)
    v_sum_but_i = get_v_sum_but_i(v)

    h = []
    for i in range(batch.shape[1]):
        batch_cur = delete_agent(batch, i)
        allocation_cur = oracle(batch_cur, language)
        v_cur = get_v(batch_cur, allocation_cur)
        v_sum_cur = v_cur.sum(dim=-1).view(-1, 1)
        h.append(v_sum_cur)
    h = torch.cat(h, dim=1)

    t = h - v_sum_but_i
    return t


if __name__ == "__main__":
    cfg = parse_cli_args()
    n_auctions, n_agents, n_items = 10000, cfg.a, cfg.i

    batch = torch.rand((n_auctions, n_agents, n_items))
    prices = -VCG(-batch, 'marginal').numpy()
    print("cost", float(prices.sum()) / n_auctions)
