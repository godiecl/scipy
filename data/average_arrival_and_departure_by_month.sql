/*
 * Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
 */

-- Average arrival and departure delays by month
SELECT month,
    ROUND(AVG(arrdelay), 2) AS avg_arrdelay,
    ROUND(AVG(depdelay), 2) AS avg_depdelay
FROM flights
GROUP BY month
ORDER BY avg_depdelay
