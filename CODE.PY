import machine
import utime

# --- CONFIGURACIÓN DE PINES (Motores 12V) ---

motor_izq1 = machine.Pin(10, machine.Pin.OUT)

motor_izq2 = machine.Pin(9, machine.Pin.OUT)

motor_der1 = machine.Pin(21, machine.Pin.OUT)

motor_der2 = machine.Pin(22, machine.Pin.OUT)

# Sensores y Botón

BOTON = machine.Pin(4, machine.Pin.IN, machine.Pin.PULL_UP)

TRIGGER = machine.Pin(17, machine.Pin.OUT)

ECHO = machine.Pin(16, machine.Pin.IN)



modo_combate = False



# --- FUNCIONES DE MOVIMIENTO ---

def parado():
    motor_izq1.low(); motor_izq2.low()
    motor_der1.low(); motor_der2.low()



def adelante():
    motor_izq1.high(); motor_izq2.low()
    motor_der1.high(); motor_der2.low()



def girar_busqueda():

    # Gira sobre su propio eje para barrer el área con el sensor

    motor_izq1.high(); motor_izq2.low()

    motor_der1.low(); motor_der2.high()



def obtener_distancia():

    TRIGGER.low()

    utime.sleep_us(2)

    TRIGGER.high()

    utime.sleep_us(10)

    TRIGGER.low()

    duracion = machine.time_pulse_us(ECHO, 1, 30000)

    if duracion < 0: return 0

    return (duracion * 0.0343) / 2



# --- BUCLE PRINCIPAL ---

print("Esperando botón de inicio...")



while True:

    # 1. LÓGICA DEL BOTÓN Y CONTEO

    if BOTON.value() == 0:

        utime.sleep_ms(200) # Anti-rebote

        print("¡Iniciando secuencia de 5 segundos!")

       

        # EL RETROCESO (REGLA DE ORO)

        for i in range(5, 0, -1):

            print(f"Combate inicia en... {i}")

            parado() # Se asegura de que no se mueva

            utime.sleep(1)

       

        modo_combate = True

        print("¡¡ROBETO AL ATAQUE!!")



    # 2. LÓGICA DE COMBATE

    if modo_combate:

        dist = obtener_distancia()

       

        # Si detecta algo a menos de 40cm (ajustable)

        if 0 < dist < 40:

            print(f" ¡EMBISTIENDO!")

            adelante()

        else:

            # Si no ve nada, gira para buscarlo

            print("Buscando oponente...")

            girar_busqueda()

           

    utime.sleep_ms(50) # Pausa para estabilidad del sensor
