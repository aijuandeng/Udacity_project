import time
import pandas as pd
import numpy as np

CITY_DATA = {'chicago': 'chicago.csv',
             'new york city': 'new_york_city.csv',
             'washington': 'washington.csv'}
cities = ("chicago", "new york city", "washington")
months = ("all", "Janury", "February", "March", "April", "May", "June")
days = ("all", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day

        - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    try:
        city = input('which city would you want to analyze?(chicago, new york city, washington)\n')
        while city.lower().strip() not in cities:
            city = input('sorry!i only know three cityies:chicago, new york city, washington;please input one of them\n')#the invalid inputs will ba hinted
        else:
            month = input('which month would you want to analyze?or all?(all,Janury, February, March, April, May, June)\n')
            while month.strip().capitalize() not in months:
                month = input('oh sorry?please retype eg:all,Janury, February, March, April, May, June)\n')
            else:
                day = input(
                    'which day would you want to analyze?or all?(all,Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)\n')
                while day.strip().capitalize() not in days:
                    day = input(
                        'sorry!you need to retype,because i only know: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday;please input one of them)\n')
                else:
                    city = city.strip().lower()
                    month = month.strip().capitalize()
                    day = day.strip().capitalize()
                    return city, month, day
    except Exception as e:
        print(e)

def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.
    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    df = pd.read_csv(CITY_DATA[city])
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.weekday_name
    df["hour"] = df['Start Time'].dt.hour
    if month != 'all':
        month = months.index(month)
        df = df[df['month'] == month]
    if day != 'all':
        df = df[df['day_of_week'] == day]
    return df

def time_stats(df):
    """Displays statistics on the most frequent times of travel."""
    start_time = time.time()
    print(df['month'])
    month_mode = df["month"].mode()# [0]
    print('1',month_mode)
    day_of_week_mode = df['day_of_week'].mode()[0]
    print('2',day_of_week_mode)
    hour_mode = df["hour"].mode()[0]
    print('3',hour_mode)
    df_one = pd.DataFrame(data=[[month_mode, day_of_week_mode, hour_mode]],
                            columns=['most Frequent month|', 'most common day|', 'most common start hour|'])#building a dataframe to Information Aggregation
    print('it shows the most common travel date in this city')
    print(df_one)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-' * 40)

def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    start_time = time.time()
    Start_Station = df["Start Station"].value_counts().idxmax()
    End_Station = df["End Station"].value_counts().idxmax()
    list_start_station = list(df['Start Station'])
    list_end_station = list(df['End Station'])
    #i try to use zip to combination them and build a series to count
    tuple_station = tuple(zip(list_start_station, list_end_station))
    most_frequent_line = pd.Series(tuple_station).mode()
    df_two = pd.DataFrame(data=[[Start_Station, End_Station, most_frequent_line]],
                            columns=['most_commonly_Start_Station|', 'most_commonly_End_Station|', 'most_frequent_line|'])
    print('there is an analysis of the most frequent Stations and Trip')
    print(df_two)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-' * 40)

def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    start_time = time.time()
    df["Trip Duration"] = df["Trip Duration"]/60
    Trip_Duration_mode = df["Trip Duration"].mode()[0]
    Trip_Duration_sum = df["Trip Duration"].sum()
    Trip_Duration_mean = df["Trip Duration"].mean()
    df_three = pd.DataFrame(data=[[Trip_Duration_mode, Trip_Duration_sum, Trip_Duration_mean]],
                            columns=['most_frequent_travel_time|', 'total_travel_time|', 'mean_travel_time|'])
    print('there is an analysis of the travel time,in minutes')
    print(df_three)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-' * 40)

def user_stats(df,city):
    """Displays statistics on bikeshare users."""
    start_time = time.time()
    if city.lower() != "washington":
        user_types = df['User Type'].value_counts()
        most_recent_birth = int(df["Birth Year"].max())
        earliest_birth = int(df["Birth Year"].min())
        most_common_birth = int(df["Birth Year"].mode()[0])
        print('there is an summary of the user types')
        print(user_types)
        df_four = pd.DataFrame(data=[[most_recent_birth, earliest_birth, most_common_birth]],
                            columns=['most_recent_birth|', 'earliest_birth|', 'most_common_birth|'])
        print('there is an analysis of the user year of birth')
        print(df_four)
        print("\nThis took %s seconds." % (time.time() - start_time))
        print('-' * 40)
    else:
        pass

def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)
        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df, city)
        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower().strip() != 'yes':
            break
        else:
            pass

if __name__ == '__main__':
    main()
