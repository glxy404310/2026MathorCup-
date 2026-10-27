"""Fixed-tree consensus follow-up after unsuccessful LOSO domain transfer.

The domain experiment remains an unsuccessful recorded experiment. This
follow-up isolates joint screening at the previous fixed-tree protocol;
its defaults are fixed before any of its new audit draws are evaluated.
"""
from importlib import import_module

from threadpoolctl import threadpool_limits

from common import SOLUTION


if __name__=="__main__":
    transfer=import_module("17_q3_transfer_domain")
    with threadpool_limits(limits=1):
        transfer.main(output_directory=SOLUTION/"outputs/q3_domain/consensus",
            proxy_directory=SOLUTION/"outputs/q3_optimized/stale_fixed",
            experiment="fixed_tree_consensus_followup",
            fresh_domain_audit_seed=20290123,
            fresh_original_audit_seed=20291123)
