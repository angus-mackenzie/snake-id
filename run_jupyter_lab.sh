#!/bin/bash

# need to specify ip of computer, otherwise cant access jupyter remotely...
jupyter-lab --ip "$(hostname -I | awk '{print $1}')"
