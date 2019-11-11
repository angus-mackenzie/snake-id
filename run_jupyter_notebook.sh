#!/bin/bash

# need to specify ip of computer, otherwise cant access jupyter remotely...
# WARNING: this is not secure
jupyter-notebook --ip "$(hostname -I | awk '{print $1}')"
