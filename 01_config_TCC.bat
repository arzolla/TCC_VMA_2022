@echo off

python .\config.py -m Town04
echo Weathers disponiveis:
echo/
 echo   ClearNight, ClearNoon, ClearSunset, CloudyNight, CloudyNoon,
 echo   CloudySunset, Default, HardRainNight, HardRainNoon,
 echo   HardRainSunset, MidRainSunset, MidRainyNight, MidRainyNoon,
 echo   SoftRainNight, SoftRainNoon, SoftRainSunset, WetCloudyNight,
 echo   WetCloudyNoon, WetCloudySunset, WetNight, WetNoon, WetSunset.
 echo/
set /p weather="Selecione o weather: "


python .\config.py --weather %weather%
echo