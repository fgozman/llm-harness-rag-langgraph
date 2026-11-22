# Servo Tuning Guide

This guide covers PID tuning for Northwind joint controllers (firmware 4.2 and later).

## Prerequisites

- Firmware 4.2 or later on every joint controller.
- The arm must be unloaded and in the home position.

## Default gains

The factory defaults are a good starting point for most installations.

| Joint | Kp | Ki | Kd |
|-------|-----|------|------|
| J1 | 120 | 0.80 | 4.5 |
| J2 | 140 | 0.95 | 5.1 |
| J3 | 95 | 0.60 | 3.2 |
| J4 | 60 | 0.35 | 1.9 |

## Procedure

### Step 1: Proportional gain

Increase Kp in steps of 10 until the joint oscillates, then back off by 30 percent.

### Step 2: Integral and derivative

Raise Ki until steady-state error is below 0.05 degrees. Add Kd only if overshoot exceeds 2 percent.

## Troubleshooting

Error code E-217 means the encoder lost its index pulse; re-seat the encoder cable on the affected joint.
