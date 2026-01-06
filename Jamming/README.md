# Jamming

## Overview

**Jamming** is an electronic warfare technique used to intentionally disrupt or deny wireless communication by transmitting interference signals on the same frequency band as the target system.  
In anti-drone systems, jamming is used to break the communication link between a drone and its controller or to interfere with navigation signals (such as GNSS), forcing the drone to hover, land, or return to home.

---

## Types of Jamming

### Spot Jamming
Spot jamming concentrates all the jamming power on **a single frequency or a very narrow frequency band**.

- Highly effective when the target frequency is known  
- Power-efficient due to focused interference  
- Limited effectiveness against frequency-hopping systems  

**Use case:** Jamming a drone control link operating on a fixed channel (e.g., a specific 2.4 GHz channel).

---

### Sweep Jamming
Sweep jamming **continuously scans across a range of frequencies**, transmitting interference sequentially.

- Covers multiple frequencies over time  
- Effective against slow channel-switching systems  
- Lower interference power per frequency  

**Use case:** Disrupting drones using basic adaptive frequency techniques.

---

### Wideband Jamming
Wideband jamming transmits interference **simultaneously across a wide frequency range**.

- Affects multiple channels at once  
- Effective against fast frequency-hopping systems  
- Requires high transmission power  

**Use case:** Blocking drone communication protocols that hop rapidly across frequencies.
