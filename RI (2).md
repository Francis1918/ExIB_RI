# Contexto de trabajo para Recuperación de la Información

Al inicio, esta carpeta contiene únicamente `RI.md`. El enunciado llegará por chat o como archivo y definirá qué construir, con qué datos y en qué formato. La implementación empieza en ese momento. Si se pide un notebook Jupyter, créalo desde cero en esta carpeta o completa el que se entregue con el ejercicio. Usa rutas relativas.

## Cómo empezar con un enunciado nuevo

1. **Define qué debe devolver el sistema.** Identifica la consulta, qué cuenta como documento y qué haría relevante un resultado. Por ejemplo, devolver cinco reseñas y devolver cinco películas distintas son tareas diferentes aunque usen el mismo CSV.
2. **Reconoce los datos disponibles.** Inspecciona los archivos y unas filas antes de elegir columnas. Localiza el texto, los identificadores y las consultas. Si hay varias tablas, comprueba qué clave las une y si la unión multiplica filas. Un enlace de película puede repetirse en muchas reseñas.
3. **Traduce lo pedido a una primera prueba.** Anota los métodos exigidos, el número de resultados y la forma de evaluación. Para un buscador, prepara el corpus, construye su representación una vez y sigue una consulta hasta el texto original de su primer resultado. Cuando ese recorrido funcione, ejecuta las demás consultas. Si falta un insumo, identifica cuál y avanza con lo que puedas comprobar.

## Antecedente del curso

En el examen anterior se construyó un buscador semántico de críticas de Rotten Tomatoes. Se unieron películas y reseñas, se tomó una muestra reproducible de **15.000 reseñas** y se representó cada una mediante el título y el texto de la crítica. Se pasó cada reseña a minúsculas, se sustituyó la puntuación por espacios y se normalizaron los espacios repetidos. Se conservaron las negaciones. `all-MiniLM-L6-v2` generó un vector de 384 dimensiones por documento.

Se codificaron ocho consultas en inglés con el mismo modelo. Para cada una se ordenaron los documentos por similitud coseno y se mostraron los cinco primeros con posición, ID, título, fragmento y puntaje. Un resumen reunió los primeros puestos. El PCA se ajustó sobre 2.500 reseñas y se usó para dibujar documentos y consultas. El ranking se calculó con los vectores originales. Estas cifras describen ese caso concreto.

## Cómo adaptar la solución si aumenta la dificultad

| Lo que pide el enunciado | Decisión técnica que orienta el trabajo |
| --- | --- |
| Términos obligatorios, operadores booleanos o frases exactas | Un índice invertido relaciona cada término con sus documentos. Combina las listas según el operador y conserva posiciones de palabras si se exigen frases exactas. |
| Ordenar por coincidencia de términos | Para TF-IDF, ajusta el vocabulario y los pesos al corpus, transforma las consultas con ese mismo ajuste y ordena por coseno. Si se pide BM25, usa su propia puntuación. |
| Buscar por significado o comparar dos modelos | Elige un codificador adecuado al idioma si el modelo queda a elección. Para comparar MiniLM con otro modelo, como MPNet, codifica el mismo corpus y las mismas consultas por separado con cada modelo. Compara sus rankings con el mismo corte `k`. |
| Combinar búsqueda léxica y semántica, o reordenar candidatos | Para búsqueda híbrida puedes fusionar las posiciones de ambos rankings. Si se pide reordenamiento, recupera primero un conjunto de candidatos y puntúa esos pares de consulta y documento con un modelo de reordenamiento. |
| Buscar pasajes o trabajar con un corpus grande | Para pasajes, conserva el ID del documento de origen junto con el del fragmento. Si el límite es la memoria, codifica y compara por lotes, conservando los mejores resultados de cada consulta. |
| Evaluar relevancia | Los *qrels* relacionan consulta, documento y grado de relevancia. Alinea sus IDs con los del ranking. Precisión@k mide qué fracción de los primeros *k* es relevante, exhaustividad@k qué fracción de los relevantes se recuperó y nDCG@k valora el grado de relevancia y la posición. |
| Interpretar un gráfico de embeddings | Ajusta PCA sobre los documentos y usa esa misma transformación para las consultas. Reporta la varianza explicada. En el ejercicio anterior, los dos ejes recogieron solo el 5,84 %, lo que limita las conclusiones sobre cercanía en el gráfico. |

**Ejemplo de una consulta con dos condiciones.** Si se buscan películas «elogiadas por la crítica y rechazadas por el público», identifica qué dato respalda cada parte. Si existen puntuaciones de crítica y audiencia, puedes contrastarlas. Si solo existen reseñas de críticos, limita la conclusión a lo que dicen sus textos. Un puntaje alto de similitud indica cercanía a la consulta.

## Qué comprobar al entregar

Verifica que cada ID recupere el texto correcto, que los resultados estén ordenados para cada consulta y que se respete la unidad solicitada. Si comparas métodos, usa el mismo corpus, las mismas consultas y los mismos juicios de relevancia cuando estén disponibles. Si solo hay consultas, ilustra la calidad con resultados concretos y explica qué condiciones cumplen. En Jupyter, ejecuta las celdas en orden desde un kernel limpio y conserva las salidas comprobadas. Si falta el corpus o el modelo real, precisa qué parte pudiste validar.

## Referencias opcionales de código

- Implementaciones de búsqueda léxica, búsqueda densa y evaluación con juicios de relevancia en el proyecto grupal [FiQA](https://github.com/Vladimirjon/fiqa-information-retrieval).
- TF-IDF, BM25 y comparación de rankings sobre libros de Gutenberg en la carpeta **03 Búsqueda Vectorial** de [Notebooks-RI](https://github.com/Almejandro04/Notebooks-RI).
