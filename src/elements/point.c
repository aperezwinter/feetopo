#include <stdio.h>
#include "../element.h"

element_t *feetopo_point_init(size_t tag, node_t *node)
{
    element_t *point = calloc(1, sizeof(element_t));
    point->tag = tag;
    point->volume = 0;
    point->area = 0;
    point->normal = NULL;
    point->type = calloc(1, sizeof(element_type_t));
    point->type->id = ELEMENT_TYPE_POINT1;
    point->type->dim = 0;
    point->type->order = 1;
    point->type->nodes = 1;
    point->type->vertices = 1;
    point->type->faces = 0;
    point->type->nodes_per_face = 0;
    point->physical_group = NULL;
    point->nodes = calloc(1, sizeof(node_t *));
    point->nodes[0] = node;
    return point;
}

void feetopo_point_print(element_t *point, bool verbose)
{
    printf("Point(%d): (%.2g,%.2g,%.2g)\n",
           point->tag,
           point->nodes[0]->r[0],
           point->nodes[0]->r[1],
           point->nodes[0]->r[2]);
}

int main()
{
    double r1[3] = {0, 0, 0};
    node_t *node1 = feetopo_node_init(1, r1);
    element_t *p1 = feetopo_point_init(1, node1);
    feetopo_point_print(p1, false);
    feetopo_element_free(p1);
    return 0;
}