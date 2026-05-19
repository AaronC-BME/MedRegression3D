import os

from medregression3d.utils.parsing import make_omegaconf_resolvers
from medregression3d.cli import main

if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    make_omegaconf_resolvers()
    main()
