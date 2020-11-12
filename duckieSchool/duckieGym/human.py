#!/usr/bin/env python3

"""
This is a custom script developed by FRANK based on duckietown
joystick script in order to allow user drive duckietown with joystick
and obtain log for further training.
"""

import argparse
import json
import sys
import cv2
import time
import gym
import numpy as np
import pyglet
import math
import logging

from log_util import Logger, SteeringToWheelVelWrapper
from log_schema import Episode, Step

from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

class HumanDriver:
    def __init__(self, env, max_episodes, max_steps, log_file=None, downscale=False):
        if not log_file:
            log_file = f"dataset.log"
            self.env = env
            self.env.reset()
            self.datagen = Logger(self.env, log_file=log_file)
            self.episode = 1
            self.max_episodes = max_episodes
            self.pwm_converter = SteeringToWheelVelWrapper()
            #! Logger setup:
            logging.basicConfig()
            logger = logging.getLogger('gym-duckietown')
            logger.setLevel(logging.WARNING)
            #! Recorder Setup:
            self.last_reward = 0
            #! Enter main event loop
            pyglet.clock.schedule_interval(
                self.update, 1.0 / self.env.unwrapped.frame_rate, self.env)
            #! Get Joystick
            # Registers joysticks and recording controls
            self.joysticks = pyglet.input.get_joysticks()
            assert self.joysticks, 'No joystick device is connected'
            self.joystick = self.joysticks[0]
            self.joystick.open()
            self.joystick.push_handlers(self.on_joybutton_press)
            pyglet.app.run()
            #! Log and exit
            datagen.close()
            self.env.close()

    def sleep_after_reset(self, seconds):
        for remaining in range(seconds, 0, -1):
            sys.stdout.write("\r")
            sys.stdout.write("{:2d} seconds remaining.".format(remaining))
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\rGO!            \n")
        return

    def playback(self):
        pass

    def image_resize(self,image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def on_key_press(self,symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            self.playback()
            self.env.reset()
            self.env.render()
            self.sleep_after_reset(5)
        elif symbol == key.PAGEUP:
            self.env.unwrapped.cam_angle[0] = 0
            self.env.render()
        elif symbol == key.ESCAPE or symbol == key.Q:
            self.env.close()
            sys.exit(0)
    
    def on_joybutton_press(self,joystick, button):
        """
        Event Handler for Controller Button Inputs
        Relevant Button Definitions:
        3 - Y - Resets Env.
        """

        # Y Button
        if button == 3:
            print('RESET')
            self.playback()

            self.env.reset()
            self.env.render()
            self.sleep_after_reset(5)

    def update(self,dt,env):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        #! Joystick no action do not record
        if round(self.joystick.z, 2) == 0.0 and round(self.joystick.y, 2) == 0.0:
            return

        #! Nominal Joystick Interpretation
        x = round(self.joystick.y, 2) * 0.9  # To ensure maximum trun/velocity ratio
        z = round(self.joystick.z, 2) * 3.0
        print(x,z)
        # #! Joystick deadband
        # if (abs(round(joystick.y, 2)) < 0.01):
        #     z = 0.0

        # if (abs(round(joystick.z, 2)) < 0.01):
        #     x = 0.0

        #! DRS enable for straight line
        if self.joystick.buttons[6]:
            x = -1.0
            z = 0.0

        action = np.array([-x, -z])
        pwm_left, pwm_right = self.pwm_converter.convert(-x, -z)

        #! GO! and get next
        # * Observation is 640x480 pixels
        (obs, reward, done, info) = self.env.step(action)

        if reward != -1000:
            print('Current Command: ', action,
                  ' speed. Score: ', reward)
            if ((reward > self.last_reward-0.02) or True):
                print('log')

                #! resize to Nvidia standard:
                obs_distorted_DS = self.image_resize(obs, width=200)

                #! ADD IMAGE-PREPROCESSING HERE!!!!!
                height, width = obs_distorted_DS.shape[:2]
                #print('Distorted return image Height: ', height,' Width: ',width)
                cropped = obs_distorted_DS[0:150, 0:200]

                # NOTICE: OpenCV changes the order of the channels !!!
                cropped_final = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)

                cv2.imshow('Whats logged', cropped_final)
                cv2.waitKey(1)

                step = Step(cropped_final, reward, action, done)
                self.datagen.log(step, info)
                self.last_reward = reward
            else:
                print('Bad Training Data! Discarding...')
                self.last_reward = reward
        else:
            print('!!!OUT OF BOUND!!!')

        if done:
            self.playback()
            self.env.reset()
            self.env.render()
            self.sleep_after_reset(5)
            return

        self.env.render()


if __name__ == '__main__':
    #! Parser sector:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default=None)
    parser.add_argument('--map-name', default='small_loop_cw')
    parser.add_argument('--draw-curve', default=False,
                        help='draw the lane following curve')
    parser.add_argument('--draw-bbox', default=False,
                        help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', default=True,
                        help='enable domain randomization')
    parser.add_argument('--playback', default=True,
                        help='enable playback after each session')
    parser.add_argument('--distortion', default=True)

    parser.add_argument('--raw-log', default=True,
                        help='enables recording high resolution raw log')
    parser.add_argument('--steps', default=1500,
                        help='number of steps to record in one batch')
    parser.add_argument("--nb-episodes", default=1200,
                        help='set the total episoded number', type=int)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--downscale", action="store_true")
    args = parser.parse_args()

    #! Start Env
    if args.env_name is None:
        env = DuckietownEnv(
            map_name="loop_pedestrians",
            max_steps=args.steps,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            distortion=args.distortion,
            accept_start_angle_deg=4,
            full_transparency=True,
        )
    else:
        env = gym.make(args.env_name)

    node = HumanDriver(env,max_episodes=args.nb_episodes, max_steps=args.steps, log_file=args.logfile, downscale = args.downscale)
