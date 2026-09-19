@echo off
cd /d "%~dp0"
echo ======================================================================
echo Launching Evacuation Simulation Visualizer...
echo ======================================================================
python main.py --scenario scenarios/bottleneck.json --play
pause
