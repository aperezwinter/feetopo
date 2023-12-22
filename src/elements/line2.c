#include <stdio.h>
#include "../element.h"

element_t *feetopo_line2_init(size_t tag, node_t **nodes)
{
    element_t *line2 = calloc(1, sizeof(element_t));
    line2->tag = tag;
    line2->volume = 0;
    line2->area = 0;
    line2->normal = NULL;
    line2->type = calloc(1, sizeof(element_type_t));
    line2->type->id = ELEMENT_TYPE_LINE2;
    line2->type->dim = 1;
    line2->type->order = 1;
    line2->type->nodes = 2;
    line2->type->vertices = 2;
    line2->type->faces = 2;
    line2->type->nodes_per_face = 1;
    line2->physical_group = NULL;
    line2->nodes = calloc(2, sizeof(node_t *));
    line2->nodes = nodes;
    return line2;
}