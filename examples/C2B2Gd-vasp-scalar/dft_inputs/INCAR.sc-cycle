SYS = C2B2Gd
ISTART   = 0
ICHARG   = 2 
LWAVE    = .FALSE. ! Si converges el U=0, se peude empezar el U=1 con el WAVECAR del scf y converge mejor
LCHARG = .TRUE.

ISMEAR   = 1 
SIGMA    = 0.02 
LASPH    = .TRUE. ! Mejora convergencia para U=!0
ENCUT    = 500
EDIFF    = 1E-5 
NELM     = 500 ! Esto es un poco grande pero igual hay que agrandarlo más incluso...
LSORBIT  = .FALSE. 
!LORBMOM = .TRUE. 
SAXIS    = 0 0 1 

LREAL    = Auto 

AMIX     = 0.2 
BMIX     = 0.0001 ! Esto hay que tocar para converger mejor
AMIX_MAG = 0.8 
BMIX_MAG = 0.0001 ! Esto hay que tocar para converger mejor
LORBIT   = 11 ! Esto lo ponemos =11 en el de bandas
LMAXMIX  = 4 ! Esto ayuda a la convergencia
