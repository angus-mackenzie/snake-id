#!/bin/bash

# need to specify ip of computer, otherwise cant access jupyter remotely...
jupyter-notebook --ip "$(hostname -I | awk '{print $1}')" --port 8080
