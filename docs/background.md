Here is your PDF document converted to a markdown format. 

***

# Senior Data Scientist - Business Case

## Background

Amenitiz offers a suite of tools designed to streamline hotel operations, including booking management, channel distribution, and website creation. To further empower hoteliers, we propose developing a price recommender system that predicts optimal pricing based on historical data.

**PriceAdvisor** aims to provide significant business value to hoteliers by enabling them to optimise their pricing strategies through a co-pilot and auto-pilot mode. By leveraging a price recommendation system, hoteliers can set optimal prices for the upcoming season, maximising Revenue Per Available Room (RevPAR).[1]

***

## Data Description

The provided file is in `amenitiz/data`. There are 4 sets of tables:

### Ds_bookings

- **id**: Is the id of each booking.
- **status**: Status of the booking. ‘Booked’ and ‘confirmed’ have the same meaning.
- **total_price**: Total price of the booking in euros. Total amount of the stay.
- **created_at**: Booking creation date.
- **cancelled_at**: Booking cancellation date.
- **source**: Where the booking comes from.
- **arrival_date**: Date of arrival from the property.
- **departure_date**: Date of departure from the property.
- **payment_method**: The method for paying for the booking.
- **cancelled_by**: The agent/person who canceled the booking.
- **hotel_id**: Unique identifier of each Amenitiz account (hotel). It’s unique for each hotelier.

### Ds_booked_rooms

- **id**: Is the id for each booked room.
- **booking_id**: Is the id of each booking.
- **total_adult**: Total adults staying in the room.
- **total_children**: Total children staying in the room.
- **room_id**: Is the unique identifier of each room in Amenitiz.
- **room_size**: Square meters of the room.
- **room_view**: Facing views of the room.
- **room_type**: Type of room.
- **total_price**: Total price in euros of each booked room. Total amount of the stay.

### Ds_hotel_location

- **id**: Unique identifier of each hotel_location.
- **hotel_id**: Unique identifier of each Amenitiz account (hotel). It’s unique for each hotelier.
- **address**: Address where the hotel is located.
- **city**: City where the hotel is located.
- **zip**: Zip code of the hotel.
- **country**: Country where the hotel is located.
- **latitude**: Google Maps latitude.
- **longitude**: Google Maps longitude.

### Ds_rooms

- **id**: Is the unique identifier of each room in Amenitiz.
- **number_of_rooms**: Number of individual rooms related to each room_id. (e.g. If a hotel has 5 suites, the suite would have a room_id with 5 as number_of_rooms.)
- **max_occupancy**: Maximum occupancy allowed by room.
- **max_adults**: Maximum number of adults allowed by room.
- **pricing_per_person**: Boolean if the room allows pricing per person or not (for the entire room).
- **events_allowed**: Boolean if events are allowed in the room.
- **pets_allowed**: Boolean if pets are allowed in the room.
- **smoking_allowed**: Boolean if smoking is allowed in the room.
- **children_allowed**: Boolean if children are allowed in the room.

***

## Your Assignment

Play with the data and tackle the problem using machine learning to build a price recommendation. The requirements are:

- Build a model that will be able to recommend optimal room prices to hoteliers for the given parameters.
- You can formulate the problem as you prefer as long as you can justify your choice and test the recommendation model using applicable metrics.
- Start with a baseline model that is more than a random pricing guess and see how much you can improve from there.
- Show how you evaluate and improve your model performance. Explain your choice of evaluation technique. Use at least one metric that tests how well your model predicts optimal prices or maximizes revenue.
- Using the provided dataset, derive additional features to demonstrate your data sense and creativity.
- What consequences does your model have on new hotel listings?
- Identify opportunities of using your model in our company. For what other purposes could it be used?
- Please submit one document and provide code and a writeup (e.g., in R Markdown or iPython Notebook).[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/119156600/bdce240f-2723-42e1-8214-30b2043ddd9a/Senior-Data-Scientist-Business-Case-3.pdf)