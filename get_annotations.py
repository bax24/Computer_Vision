from cytomine import Cytomine
from cytomine.models import *
from cytomine.models.image import SliceInstanceCollection
from shapely import wkt
from shapely.affinity import affine_transform

host = "https://learn.cytomine.be"
public_key = 'c4bf8472-a983-4d57-85e8-757d81afe3e6'
private_key = 'f64ddf37-8574-4d23-93a0-cf8ecdc19166'
conn = Cytomine.connect(host, public_key, private_key)

# ... Connect to Cytomine (same as previously) ...
projects = ProjectCollection().fetch()

for project in projects:

    print('### GROUP: {} ###'.format(project.name))

    images = ImageInstanceCollection().fetch_with_filter("project", project.id)
    terms = TermCollection().fetch_with_filter("project", project.id)
    slices = SliceInstanceCollection().fetch_with_filter("imageinstance", images[0].id)

    annotations = AnnotationCollection()
    annotations.showWKT = True
    annotations.showMeta = True
    annotations.showTerm = True
    annotations.showTrack = True
    annotations.showImage = True
    annotations.showSlice = True
    annotations.project = project.id

    annotations.fetch()

    file = open('annotations.txt', 'w')
    for annot in annotations:

        print("ID: {} | Image: {} | Project: {} | Terms: {} | Track: {} | Slice: {}".format(
            annot.id, annot.image, annot.project, terms.find_by_attribute("id", annot.term[0]), annot.track,
            annot.time))

        file.write("ID: {} | Image: {} | Project: {} | Terms: {} | Track: {} | Slice: {}".format(
            annot.id, annot.image, annot.project, terms.find_by_attribute("id", annot.term[0]), annot.track,
            annot.time))
        file.write('\n')

        geometry = wkt.loads(annot.location)
        # print("Geometry from Shapely (cartesian coordinate system): {}".format(geometry))

        # In OpenCV, the y-axis is reversed for points.
        # x' = ax + by + x_off => x' = x
        # y' = dx + ey + y_off => y' = -y + image.height
        # matrix = [a, b, d, e, x_off, y_off]
        image = images.find_by_attribute("id", annot.image)
        geometry_opencv = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])
        print(geometry_opencv)

        file.write(format(geometry_opencv))
        file.write('\n')


        # print("Geometry with OpenCV coordinate system: {}".format(geometry_opencv))
    file.close()