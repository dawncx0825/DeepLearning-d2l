import os
import matplotlib
matplotlib.use('Agg')

import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1, fair_probs).sample())

print(multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()

output_path = os.path.join(os.path.dirname(__file__), 'probability.png')
d2l.plt.tight_layout()
d2l.plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Figure saved to {output_path}')
d2l.plt.show()