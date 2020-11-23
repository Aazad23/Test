import numpy as np
import cv2
import math


class Snake:

    MIN_DISTANCE_BETWEEN_POINTS = 5    
    MAX_DISTANCE_BETWEEN_POINTS = 50    
    SEARCH_KERNEL_SIZE = 7             

    image = None        
    gray = None         
    binary = None       
    gradientX = None    
    gradientY = None    
    blur = None
    width = -1          
    height = -1         
    points = None       
    n_starting_points = 50       
    snake_length = 0    
    closed = True       
    alpha = 0.5         
    beta = 0.5          
    delta = 0.1         
    w_line = 0.5        
    w_edge = 0.5        
    w_term = 0.5        



    def __init__( self, image = None, closed = True ):

        self.image = image

        self.width = image.shape[1]
        self.height = image.shape[0]

        self.gray = image[:,:,0]
        blur = cv2.GaussianBlur(self.gray,(3,3),0)
        ret3,thrshld = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thrshld,kernel,iterations = 1)
        self.binary = cv2.dilate(erosion,kernel,iterations = 1)
        self.gradientX = cv2.Sobel( self.binary, cv2.CV_64F, 1, 0, ksize=5 )
        self.gradientY = cv2.Sobel( self.binary, cv2.CV_64F, 0, 1, ksize=5 )

        # Set the snake behaviour (closed or open)
        self.closed = closed

        # Gets the half width and height of the image
        # to use as center of the generated snake circle
        half_width = math.floor( self.width / 2 )
        half_height = math.floor( self.height / 2 )


        # Generating the starting points

        # If it is a closed snake, my initial guess will be a circle that covers the maximum of the image
        if self.closed:
            n = self.n_starting_points
            radius = half_width if half_width < half_height else half_height
            self.points = [ np.array([
                half_width + math.floor( math.cos( 2 * math.pi / n * x ) * radius ),
                half_height + math.floor( math.sin( 2 * math.pi / n * x ) * radius ) ])
                for x in range( 0, n )
            ]
        # If it is an open snake, the initial guess will be an horizontal line
        else:   
            n = self.n_starting_points
            factor = math.floor( half_width / (self.n_starting_points-1) )
            self.points = [ np.array([ math.floor( half_width / 2 ) + x * factor, half_height ])
                for x in range( 0, n )
            ]



    def visualize( self ):

        img = self.image.copy()

        point_color = ( 0, 255, 255 )     
        line_color = ( 128, 0, 0 )      
        thickness = 2                   

        n_points = len( self.points )
        for i in range( 0, n_points - 1 ):
            cv2.line( img, tuple( self.points[ i ] ), tuple( self.points[ i + 1 ] ), line_color, thickness )

        if self.closed:
            cv2.line(img, tuple( self.points[ 0 ] ), tuple( self.points[ n_points-1 ] ), line_color, thickness )

        [ cv2.circle( img, tuple( x ), thickness, point_color, -1) for x in self.points ]

        return img


    def dist(a, b):
        return np.sqrt( np.sum( ( a - b ) ** 2 ) )

    def normalize(kernel):
        abs_sum = np.sum( [ abs( x ) for x in kernel ] )
        return kernel / abs_sum if abs_sum != 0 else kernel

    def get_length(self):
        n_points = len(self.points)
        if not self.closed:
            n_points -= 1
        return np.sum( [ Snake.dist( self.points[i], self.points[ (i+1)%n_points  ] ) for i in range( 0, n_points ) ] )



    def f_uniformity( self, p, prev ):
        avg_dist = self.snake_length / len( self.points )
        un = Snake.dist( prev, p )
        dun = abs( un - avg_dist )
        return dun**2



    def f_curvature( self, p, prev, next ):
        ux = p[0] - prev[0]
        uy = p[1] - prev[1]
        un = math.sqrt( ux**2 + uy**2 )

        vx = p[0] - next[0]
        vy = p[1] - next[1]
        vn = math.sqrt( vx**2 + vy**2 )

        if un == 0 or vn == 0:
            return 0

        cx = float( vx + ux )  / ( un * vn )
        cy = float( vy + uy ) / ( un * vn )

        cn = cx**2 + cy**2

        return cn



    def f_line( self, p ):
        # line energy (The tendency to move the curve towards dark / lighter areas)
        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            return np.finfo(np.float64).max
        return self.binary[ p[1] ][ p[0] ]



    def f_edge( self, p ):
        # The edge energy (The tendency to move the curve towards edges). Using the sobel gradient.
        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            return np.finfo(np.float64).max
        return -( self.gradientX[ p[1] ][ p[0] ]**2 + self.gradientY[ p[1] ][ p[0] ]**2  )



    def f_term( self, p, prev, next ):
        # tendency to move the snake towards corners and terminations.
        return 0



    def f_conf( self, p , prev, next ):
        import random
        return random.random()



    def remove_overlaping_points( self ):

        snake_size = len( self.points )

        for i in range( 0, snake_size ):
            for j in range( snake_size-1, i+1, -1 ):
                if i == j:
                    continue

                curr = self.points[ i ]
                end = self.points[ j ]

                dist = Snake.dist( curr, end )

                if dist < self.MIN_DISTANCE_BETWEEN_POINTS:
                    remove_indexes = range( i+1, j ) if (i!=0 and j!=snake_size-1) else [j]
                    remove_size = len( remove_indexes )
                    non_remove_size = snake_size - remove_size
                    if non_remove_size > remove_size:
                        self.points = [ p for k,p in enumerate( self.points ) if k not in remove_indexes ]
                    else:
                        self.points = [ p for k,p in enumerate( self.points ) if k in remove_indexes ]
                    snake_size = len( self.points )
                    break

    def add_missing_points( self ):

        snake_size = len( self.points )
        for i in range( 0, snake_size ):
            prev = self.points[ ( i + snake_size-1 ) % snake_size ]
            curr = self.points[ i ]
            next = self.points[ (i+1) % snake_size ]
            next2 = self.points[ (i+2) % snake_size ]

            if Snake.dist( curr, next ) > self.MAX_DISTANCE_BETWEEN_POINTS:
                # pre-computed uniform cubig b-spline for t = 0.5
                c0 = 0.125 / 6.0
                c1 = 2.875 / 6.0
                c2 = 2.875 / 6.0
                c3 = 0.125 / 6.0
                x = prev[0] * c3 + curr[0] * c2 + next[0] * c1 + next2[0] * c0
                y = prev[1] * c3 + curr[1] * c2 + next[1] * c1 + next2[1] * c0

                new_point = np.array( [ math.floor( 0.5 + x ), math.floor( 0.5 + y ) ] )

                self.points.insert( i+1, new_point )
                snake_size += 1


    def step( self ):

        changed = False

        # length of the snake
        self.snake_length = self.get_length()
        new_snake = self.points.copy()

        # kernels
        search_kernel_size = ( self.SEARCH_KERNEL_SIZE, self.SEARCH_KERNEL_SIZE )
        hks = math.floor( self.SEARCH_KERNEL_SIZE / 2 ) # half-kernel size
        e_uniformity = np.zeros( search_kernel_size )
        e_curvature = np.zeros( search_kernel_size )
        e_line = np.zeros( search_kernel_size )
        e_edge = np.zeros( search_kernel_size )
        e_term = np.zeros( search_kernel_size )
        e_conf = np.zeros( search_kernel_size )

        for i in range( 0, len( self.points ) ):
            curr = self.points[ i ]
            prev = self.points[ ( i + len( self.points )-1 ) % len( self.points ) ]
            next = self.points[ ( i + 1 ) % len( self.points ) ]

            for dx in range( -hks, hks ): 
                for dy in range( -hks, hks ):
                    p = np.array( [curr[0] + dx, curr[1] + dy] )

                    # energy functions on p
                    e_uniformity[ dx + hks ][ dy + hks ] = self.f_uniformity( p, prev )
                    e_curvature[ dx + hks ][ dy + hks ] = self.f_curvature( p, prev, next )
                    e_line[ dx + hks ][ dy + hks ] = self.f_line( p )
                    e_edge[ dx + hks ][ dy + hks ] = self.f_edge( p )
                    e_term[ dx + hks ][ dy + hks ] = self.f_term( p, prev, next )
                    e_conf[ dx + hks ][ dy + hks ] = self.f_conf( p, prev, next )

            # normalized energies
            e_uniformity = Snake.normalize( e_uniformity )
            e_curvature = Snake.normalize( e_curvature )
            e_line = Snake.normalize( e_line )
            e_edge = Snake.normalize( e_edge )
            e_term = Snake.normalize( e_term )
            e_conf = Snake.normalize( e_conf )
            e_sum = self.alpha * e_uniformity \
                    + self.beta * e_curvature \
                    + self.w_line * e_line \
                    + self.w_edge * e_edge \
                    + self.w_term * e_term \
                    + self.delta * e_conf

            emin = np.finfo(np.float64).max
            x,y = 0,0
            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    if e_sum[ dx + hks ][ dy + hks ] < emin:
                        emin = e_sum[ dx + hks ][ dy + hks ]
                        x = curr[0] + dx
                        y = curr[1] + dy
            x = 1 if x < 1 else x
            x = self.width-2 if x >= self.width-1 else x
            y = 1 if y < 1 else y
            y = self.height-2 if y >= self.height-1 else y
            if curr[0] != x or curr[1] != y:
                changed = True
            new_snake[i] = np.array( [ x, y ] )
        self.points = new_snake
        self.remove_overlaping_points()
        self.add_missing_points()
        return changed

    def set_alpha( self, x ):
        self.alpha = x / 100

    def set_beta( self, x ):
        self.beta = x / 100

    def set_delta( self, x ):
        self.delta = x / 100

    def set_w_line( self, x ):
        self.w_line = x / 100

    def set_w_edge( self, x ):
        self.w_edge = x / 100

    def set_w_term( self, x ):
        self.w_term = x / 100