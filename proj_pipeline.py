from project_utilities import *
from datetime import datetime


def proj_pipeline(model, image, my_scale, windowsize):
    input_imgs = []
    coords = []
    boxes = []
    thresholds = {0: .98,
                  1: .98,
                  2: .99,
                  3: .98,
                  4: .95,
                  5: .95,
                  6: .95,
                  7: .95,
                  8: .95,
                  9: .98}

    origin_img = image.copy()
    print('Gathering all the boxes:', datetime.now().time())
    for sizer in my_scale:
        temp_img = cv2.resize(origin_img, None, fy=sizer[0], fx=sizer[0])
        temp_img = cv2.GaussianBlur(temp_img, (5, 5), 0)

        print('  --> Window size:', temp_img.shape)
        stepsize = sizer[1]
        h, w, d = temp_img.shape
        for y in range(0, h-windowsize+stepsize, stepsize):
            for x in range(0, w-windowsize+stepsize, stepsize):
                template = temp_img[y:y+windowsize, x:x+windowsize]
                template = cv2.resize(template, (windowsize, windowsize))
                input_imgs.append(template)
                coords.append([y/sizer[0], x/sizer[0], y/sizer[0]+windowsize,
                               x/sizer[0]+windowsize, sizer[1], windowsize])

    imgs = np.asarray(input_imgs) / 255.
    print('  --> Total boxes found:', imgs.shape[0])
    print('Predicting found boxes:', datetime.now().time())
    preds = model.predict_model(imgs)
    print('Finding best boxes:', datetime.now().time())

    for i, img in enumerate(input_imgs):
        p_max = int(np.argmax(preds[i]))
        p = np.round((preds[i]) / preds[i].sum(), 3)

        if p_max != 10 and thresholds[p_max] <= p[p_max]:
            boxes.append(coords[i] + [p_max, p[p_max]])

    if len(boxes) == 0:
        print('No boxes found!')
    else:
        boxes = np.asarray(boxes)

        # Used to debug found boxes
        if False:
            all_boxes = origin_img.copy()
            for box in boxes:
                x1, y1, x2, y2, steps, windowsize, numb, perc = box
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                cv2.rectangle(all_boxes, (y1, x1), (y2, x2), (0, 0, 255), 1)
                cv2.putText(all_boxes, str(int(numb)), (y1 + 8, x1 - 1), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

            cv2.imshow('All the Boxes', all_boxes)
            cv2.imwrite('all_the_boxes.png', all_boxes)
            cv2.waitKey(0)

        print('  --> Final number of boxes:', boxes.shape[0])
        print('Non max suppression started:', datetime.now().time())
        final_boxes = box_mean_reduce(boxes)

        # Used to debug found boxes
        if False:
            all_boxes = origin_img.copy()
            for box in final_boxes:
                x1, y1, x2, y2, steps, windowsize, numb, perc = box
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                cv2.rectangle(all_boxes, (y1, x1), (y2, x2), (0, 0, 255), 1)
                cv2.putText(all_boxes, str(int(numb)), (y1 + 8, x1 - 1), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

            cv2.imshow('All the Boxes', all_boxes)
            cv2.imwrite('all_the_boxes_suppressed.png', all_boxes)
            cv2.waitKey(0)


        final_boxes = modified_non_max_suppress(final_boxes)
        if final_boxes is None or final_boxes == []:
            print('No digits found')
        else:
            print('  --> Final box counts:', final_boxes.shape[0])
            fy1 = int(np.min(final_boxes[:, 0]))
            fx1 = int(np.min(final_boxes[:, 1]))
            fy2 = int(np.max(final_boxes[:, 2]))
            fx2 = int(np.max(final_boxes[:, 3]))

            display_number = ''
            sorted_boxes = np.argsort(final_boxes, axis=0)[:, 1]
            for box in sorted_boxes:
                x1, y1, x2, y2, steps, windowsize, numb, perc = final_boxes[box]
                display_number = display_number + str(int(numb))

            cv2.rectangle(origin_img, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            cv2.rectangle(origin_img, (fx1, fy1), (fx2, fy2), (0, 0, 0), 1)
            cv2.putText(origin_img, display_number, (fx1 + 24, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 4)
            cv2.putText(origin_img, display_number, (fx1 + 24, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1)

    return origin_img
