# Un agente que aprende Pong desde cero

Una red neuronal que aprende sola a jugar Pong de Atari, sin reglas ni ejemplos: solo mira los píxeles de la pantalla, prueba, y recibe +1 cuando hace un punto y −1 cuando lo pierde. Nadie le dice qué es una paleta ni una pelota. Con esa única señal, en 600 partidas pasó de hacer 2 puntos por partido a hacer unos 10.

Es aprendizaje por refuerzo (*reinforcement learning*): el agente aprende por prueba y error a elegir la acción que más recompensa le va a dar a largo plazo. Implementé a mano, en PyTorch, la Deep Q-Network (DQN) del paper de DeepMind de 2015, la que aprendió a jugar decenas de juegos de Atari, para entender cada pieza por dentro.

Para verlo jugar: [partida 1](videos/partida-001.mp4), donde todavía mueve la paleta al azar, y [partida 200](videos/partida-200.mp4), que perdió 21 a 6 cuando aún elegía al azar un 37 % de las veces.

## Resultados

Un partido de Pong termina cuando alguien llega a 21. La recompensa es la diferencia: −21 es perder 21 a 0, 0 es empatar el ritmo, +21 es ganar 21 a 0.

| Tanda | Partidas | Promedio | Mejor partida | Exploración al final (epsilon) |
|---|---|---|---|---|
| 1 | 1–200 | −18,7 | −12 | 0,37 |
| 2 | 201–250 | −16,8 | −12 | 0,29 |
| 3 | 251–300 | −15,9 | −11 | 0,22 |
| 4 | 301–350 | −16,0 | −10 | 0,17 |
| 5 | 351–400 | −15,2 | −9 | 0,14 |
| 6 | 401–450 | −13,3 | −7 | 0,11 |
| 7 | 451–500 | −12,3 | −3 | 0,08 |
| 8 | 501–550 | −11,9 | −6 | 0,06 |
| 9 | 551–600 | −10,8 | −3 | 0,05 |

La mejor partida terminó 21 a 18. **Todavía no le gana a la computadora.** Fueron unos 1.000.000 de pasos en una laptop, con una memoria de jugadas 100 veces más chica que la del paper, y esa memoria arranca vacía en cada tanda (solo se guarda la red). La curva sigue subiendo, así que más tandas deberían mejorarlo. Los registros completos de cada tanda están en [`registros/`](registros/).

En [`videos/`](videos/) hay cuatro partidas de la primera tanda (la 1, la 50, la 100 y la 200), grabadas mientras el agente todavía exploraba mucho al azar.

## Cómo funciona

- **Lo que ve:** cada imagen se pasa a gris y se reduce a 84×84. Se apilan los últimos 4 cuadros para que el agente pueda deducir hacia dónde va la pelota.
- **La red:** tres capas convolucionales y dos densas (la arquitectura del paper). A la salida da un valor por cada acción posible: cuánto espera ganar si la elige.
- **Explorar o aprovechar (epsilon):** al principio elige casi siempre al azar. Después de cada partida el azar baja un 0,5 %, hasta quedar en 5 %.
- **Memoria de jugadas:** guarda las últimas 10.000 jugadas y aprende de lotes de 32 tomados al azar, para no aprender solo de lo que acaba de pasar.
- **Red objetivo:** una copia de la red que se actualiza cada 1.000 pasos y estabiliza el aprendizaje.
- **Ecuación de Bellman:** el valor de una jugada es lo que ganó ahora más el 99 % de lo mejor que espera ganar después.

## Cómo correrlo

Necesita Python 3.10 o más nuevo.

```
pip install -r requirements.txt
python entrenar_pong.py
```

Por defecto juega 50 partidas y graba una cada 10 en `video_pong_local/`. Guarda lo aprendido en `cerebro_pong.pth` cada 25 partidas y al terminar; si volvés a correrlo, sigue desde ahí. Las opciones:

```
python entrenar_pong.py --partidas 200 --video-cada 50
```

Con GPU (CUDA) la usa sola. En CPU, una tanda de 50 partidas tarda horas.

## Nota sobre el código

El script de este repo es la versión ordenada del que generó los registros. Tiene un cambio: cuando una partida se corta por límite de tiempo (y no porque alguien llegó a 21), ya no se la trata como un final, porque el partido podría haber seguido. En Pong casi nunca pasa, así que no cambia los resultados de arriba.

## Referencias

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518, 529–533.
- Morales, M. (2020). *Grokking Deep Reinforcement Learning*. Manning, cap. 8 y 9.
- [Gymnasium](https://gymnasium.farama.org/) y el [Arcade Learning Environment](https://ale.farama.org/).

## Licencia

MIT. Ver [LICENSE](LICENSE).
