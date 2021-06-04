#!/bin/bash
docker exec -itd patch_transformer tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
